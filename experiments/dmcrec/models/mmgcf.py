import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from cornac.models import Recommender

class MMGCFTorch(nn.Module):
    def __init__(self, n_users, n_items, interaction_matrix, v_feat, t_feat, config):
        super(MMGCFTorch, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items

        device_arg = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_arg) if isinstance(device_arg, str) else device_arg

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_ui_layers = config['n_ui_layers']
        self.fusion_mode = config['fusion_mode']
        self.weighting = config['weighting']
        self.dropout = config['dropout']

        self.interaction_matrix = interaction_matrix.tocoo().astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.n_modalities = 0
        if v_feat is not None:
            v_feat_tensor = torch.FloatTensor(v_feat)
            self.image_embedding = nn.Embedding.from_pretrained(v_feat_tensor, freeze=True)
            self.image_trs = nn.Linear(v_feat.shape[1], self.feat_embed_dim)
            self.n_modalities += 1
        else:
            self.image_embedding = None

        if t_feat is not None:
            t_feat_tensor = torch.FloatTensor(t_feat)
            self.text_embedding = nn.Embedding.from_pretrained(t_feat_tensor, freeze=True)
            self.text_trs = nn.Linear(t_feat.shape[1], self.feat_embed_dim)
            self.n_modalities += 1
        else:
            self.text_embedding = None

        if self.weighting == 'alpha':
            self.mm_alpha = nn.Parameter(torch.tensor(0.0))

        if self.fusion_mode == 'concat' and self.n_modalities > 0:
            all_in = self.embedding_dim + self.n_modalities * self.feat_embed_dim
            self.all_concat_layer = nn.Linear(all_in, self.embedding_dim)

            if self.n_modalities > 1:
                mm_in = self.n_modalities * self.feat_embed_dim
                self.mm_concat_layer = nn.Linear(mm_in, self.feat_embed_dim)

            id_mm_in = self.embedding_dim + self.feat_embed_dim
            self.id_mm_concat_layer = nn.Linear(id_mm_in, self.embedding_dim)

        self.to(self.device)

    def get_norm_adj_mat(self):
        R = self.interaction_matrix.tocsr()
        n_users = self.n_users
        n_items = self.n_nodes - self.n_users

        top_block = sp.hstack([sp.csr_matrix((n_users, n_users), dtype=np.float32), R])
        bottom_block = sp.hstack([R.T, sp.csr_matrix((n_items, n_items), dtype=np.float32)])
        A = sp.vstack([top_block, bottom_block], format="coo")

        deg = np.array(A.sum(axis=1)).flatten()
        deg[deg == 0] = 1e-12
        d_inv = np.power(deg, -0.5)

        values = d_inv[A.row] * A.data * d_inv[A.col]  # type: ignore

        indices = torch.LongTensor(np.vstack((A.row, A.col)))  # type: ignore
        data = torch.FloatTensor(values)

        with torch.sparse.check_sparse_tensor_invariants():
            ret = torch.sparse_coo_tensor(indices, data, (self.n_nodes, self.n_nodes)).to(self.device)

        del A, R, top_block, bottom_block, deg, d_inv, values, indices, data
        return ret

    def _normalize_adj_m(self, indices):
        with torch.no_grad():
            u_idx = indices[0].to(self.device, dtype=torch.long)
            i_idx = indices[1].to(self.device, dtype=torch.long)
            num_edges = indices.shape[1]

            u_deg = torch.zeros(self.n_users, device=self.device, dtype=torch.float32)
            i_deg = torch.zeros(self.n_items, device=self.device, dtype=torch.float32)

            u_deg.scatter_add_(0, u_idx, torch.ones(num_edges, device=self.device, dtype=torch.float32))
            i_deg.scatter_add_(0, i_idx, torch.ones(num_edges, device=self.device, dtype=torch.float32))

            u_deg += 1e-7
            i_deg += 1e-7

            # D_u^(-1/2) * D_i^(-1/2)
            vals = torch.pow(u_deg[u_idx], -0.5) * torch.pow(i_deg[i_idx], -0.5)

            del u_deg, i_deg, u_idx, i_idx
            return vals

    def get_edge_info(self):
        with torch.no_grad():
            coo_mat = self.interaction_matrix.tocoo()

            rows = torch.from_numpy(coo_mat.row.astype(np.int64)).to(self.device)
            cols = torch.from_numpy(coo_mat.col.astype(np.int64)).to(self.device)
            edges = torch.stack([rows, cols], dim=0)

            vals = self._normalize_adj_m(edges)

            del rows, cols, coo_mat
            return edges, vals

    def pre_epoch_processing(self):
        if self.dropout <= 0.0:
            self.masked_adj = self.norm_adj
            return

        with torch.no_grad():
            degree_len = int(self.edge_values.size(0) * (1.0 - self.dropout))
            degree_idx = torch.multinomial(self.edge_values.to(self.device), degree_len, replacement=False)

            keep_idx = self.edge_indices[:, degree_idx].clone().to(self.device)
            keep_vals = self._normalize_adj_m(keep_idx)  # type: ignore
            all_vals = torch.cat((keep_vals, keep_vals))

            keep_idx_mapped = keep_idx.clone()
            keep_idx_mapped[1] += self.n_users
            all_idx = torch.cat((keep_idx_mapped, torch.flip(keep_idx_mapped, [0])), dim=1)

            with torch.sparse.check_sparse_tensor_invariants():
                self.masked_adj = torch.sparse_coo_tensor(
                    all_idx, all_vals, self.norm_adj.shape
                )

            del degree_idx, keep_idx, keep_vals, all_vals, keep_idx_mapped, all_idx
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def lightgcn_propagate(self, adj):
        ego = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        out = ego.clone()
        for _ in range(self.n_ui_layers):
            ego = torch.sparse.mm(adj, ego)
            out = out + ego
        out = out / (self.n_ui_layers + 1.0)
        return torch.split(out, [self.n_users, self.n_items], dim=0)

    def _get_mm_feats(self):
        feats = []
        if self.image_embedding is not None:
            feats.append(self.image_trs(self.image_embedding.weight))
        if self.text_embedding is not None:
            feats.append(self.text_trs(self.text_embedding.weight))
        return feats

    def _apply_fusion(self, tensors, concat_layer=None):
        if self.fusion_mode == 'mean':
            res = tensors[0]
            for t in tensors[1:]:
                res = res + t
            return res / len(tensors)
        elif self.fusion_mode == 'sum':
            res = tensors[0]
            for t in tensors[1:]:
                res = res + t
            return res
        else: # concat
            return concat_layer(torch.cat(tensors, dim=-1))  # type: ignore

    def fuse_item_embeddings(self, item_emb):
        mm_feats = self._get_mm_feats()
        if not mm_feats:
            return item_emb

        if self.weighting == 'alpha':
            alpha = torch.sigmoid(self.mm_alpha)
            tensors = [item_emb * alpha] + [f * (1.0 - alpha) for f in mm_feats]
            return self._apply_fusion(tensors, concat_layer=self.all_concat_layer if self.fusion_mode == 'concat' else None)

        elif self.weighting == 'normalized':
            tensors = [F.normalize(item_emb, p=2, dim=-1) * self.n_modalities] + [F.normalize(f, p=2, dim=-1) for f in mm_feats]
            return self._apply_fusion(tensors, concat_layer=self.all_concat_layer if self.fusion_mode == 'concat' else None)

        else: # equal
            if len(mm_feats) > 1:
                mm_fused = self._apply_fusion(mm_feats, concat_layer=self.mm_concat_layer if self.fusion_mode == 'concat' else None)
            else:
                mm_fused = mm_feats[0]
            return self._apply_fusion([item_emb, mm_fused], concat_layer=self.id_mm_concat_layer if self.fusion_mode == 'concat' else None)

    def forward(self, adj):
        user_emb, item_emb = self.lightgcn_propagate(adj)
        item_emb = self.fuse_item_embeddings(item_emb)
        return user_emb, item_emb


class MMGCF(Recommender):
    def __init__(
        self,
        name="MMGCF",
        embedding_size=64,
        feat_embed_dim=64,
        n_ui_layers=3,
        reg_weight=1e-4,
        fusion_mode="concat",
        weighting="equal",
        dropout=0.2,
        n_epochs=50,
        batch_size=512,
        learning_rate=0.001,
        device=None,
        trainable=True,
        verbose=False,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.config = {
            "embedding_size": embedding_size,
            "feat_embed_dim": feat_embed_dim,
            "n_ui_layers": n_ui_layers,
            "fusion_mode": fusion_mode,
            "weighting": weighting,
            "dropout": dropout,
            "device": torch.device(device) if isinstance(device, str) else device
        }
        self.reg_weight = reg_weight
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = learning_rate

        self.torch_model = None

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        v_feat, t_feat = None, None

        if train_set.item_image is not None:
            raw_v_feat = train_set.item_image.features[:train_set.num_items]
            v_feat = raw_v_feat / (np.linalg.norm(raw_v_feat, axis=1, keepdims=True) + 1e-7)

        if train_set.item_text is not None:
            raw_t_feat = train_set.item_text.features[:train_set.num_items]
            t_feat = raw_t_feat / (np.linalg.norm(raw_t_feat, axis=1, keepdims=True) + 1e-7)

        interaction_matrix = train_set.matrix

        self.torch_model = MMGCFTorch(
            n_users=train_set.num_users,
            n_items=train_set.num_items,
            interaction_matrix=interaction_matrix,
            v_feat=v_feat,
            t_feat=t_feat,
            config=self.config
        )

        optimizer = optim.Adam(self.torch_model.parameters(), lr=self.lr)
        device = self.torch_model.device


        from tqdm.auto import tqdm

        for epoch in range(1, self.n_epochs + 1):
            self.torch_model.train()
            self.torch_model.pre_epoch_processing()

            sum_loss = 0.0
            count = 0

            progress_bar = tqdm(
                total=train_set.num_batches(self.batch_size),
                desc=f"Epoch {epoch}/{self.n_epochs}",
                disable=not self.verbose,
            )

            for batch_users, batch_pos_items, batch_neg_items in train_set.uij_iter(
                batch_size=self.batch_size, shuffle=True
            ):
                optimizer.zero_grad()

                u_t = torch.tensor(batch_users, dtype=torch.long, device=device)
                pos_t = torch.tensor(batch_pos_items, dtype=torch.long, device=device)
                neg_t = torch.tensor(batch_neg_items, dtype=torch.long, device=device)

                ua_emb, ia_emb = self.torch_model(self.torch_model.masked_adj)

                pos_scores = (ua_emb[u_t] * ia_emb[pos_t]).sum(dim=1)
                neg_scores = (ua_emb[u_t] * ia_emb[neg_t]).sum(dim=1)
                mf_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

                reg_loss = (
                    self.torch_model.user_embedding.weight[u_t].norm(2).pow(2)
                    + self.torch_model.item_id_embedding.weight[pos_t].norm(2).pow(2)
                    + self.torch_model.item_id_embedding.weight[neg_t].norm(2).pow(2)
                ) / (2 * len(batch_users))

                loss = mf_loss + self.reg_weight * reg_loss
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                count += len(batch_users)
                progress_bar.set_postfix(loss=sum_loss / count * 100)
                progress_bar.update(1)
            progress_bar.close()

        self.torch_model.eval()
        with torch.no_grad():
            u_emb, i_emb = self.torch_model(self.torch_model.norm_adj)
            self.user_emb_final = u_emb.cpu().numpy()
            self.item_emb_final = i_emb.cpu().numpy()
        return self

    def score(self, user_idx, item_idx=None):
        u_emb = self.user_emb_final[user_idx]
        if item_idx is None:
            return np.dot(u_emb, self.item_emb_final.T)
        else:
            return np.dot(u_emb, self.item_emb_final[item_idx].T)
