import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cornac.models import Recommender


class RawLATTICE(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        norm_adj,
        embedding_size,
        feat_embed_dim,
        knn_k,
        lambda_coeff,
        n_layers,
        reg_weight,
        v_feat=None,
        t_feat=None,
        device="cpu",
    ):
        super(RawLATTICE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_size
        self.feat_embed_dim = feat_embed_dim
        self.knn_k = knn_k
        self.lambda_coeff = lambda_coeff
        self.n_layers = n_layers
        self.n_ui_layers = 2
        self.reg_weight = reg_weight
        self.device = torch.device(device) if isinstance(device, str) else device

        self.norm_adj = norm_adj

        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.v_feat = v_feat
        self.t_feat = t_feat
        self.num_modal = 0

        if self.v_feat is not None:
            v_feat_norm = F.normalize(self.v_feat, p=2, dim=1)
            self.image_embedding = nn.Embedding.from_pretrained(
                v_feat_norm, freeze=True
            )
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.image_trs.weight)
            self.num_modal += 1

            self.image_original_adj = self.build_knn_laplacian_sparse(self.image_embedding.weight, topk=self.knn_k)

        if self.t_feat is not None:
            t_feat_norm = F.normalize(self.t_feat, p=2, dim=1)
            self.text_embedding = nn.Embedding.from_pretrained(
                t_feat_norm, freeze=True
            )
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.text_trs.weight)
            self.num_modal += 1

            self.text_original_adj = self.build_knn_laplacian_sparse(self.text_embedding.weight, topk=self.knn_k)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

        self.item_adj = None

    def _get_topk_chunked(self, feats, topk, chunk_size=4096):
        with torch.no_grad():
            N = feats.size(0)
            norm_feats = F.normalize(feats, p=2, dim=-1)

            topk_vals_list, topk_ind_list = [], []

            for i in range(0, N, chunk_size):
                chunk = norm_feats[i : i + chunk_size]
                sim_chunk = torch.mm(chunk, norm_feats.t())

                topk_vals, topk_ind = torch.topk(sim_chunk, k=topk, dim=-1)
                topk_vals_list.append(topk_vals)
                topk_ind_list.append(topk_ind)

                del chunk, sim_chunk

            topk_sim = torch.cat(topk_vals_list, dim=0)
            knn_ind = torch.cat(topk_ind_list, dim=0)

            row = torch.arange(N, device=self.device).view(-1, 1).expand(-1, topk).flatten()
            col = knn_ind.flatten()
            vals = topk_sim.flatten()

            sym_row = torch.cat([row, col], dim=0)
            sym_col = torch.cat([col, row], dim=0)
            sym_vals = torch.cat([vals * 0.5, vals * 0.5], dim=0)

            indices = torch.stack([sym_row, sym_col], dim=0)

            deg = torch.zeros(N, device=self.device, dtype=torch.float32)
            deg.scatter_add_(0, sym_row, sym_vals)

            deg_inv_sqrt = torch.pow(deg + 1e-7, -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

            norm_vals = deg_inv_sqrt[sym_row] * sym_vals * deg_inv_sqrt[sym_col]

            del norm_feats, topk_vals_list, topk_ind_list, topk_sim, knn_ind, row, col, vals, sym_row, sym_col, sym_vals, deg, deg_inv_sqrt
            return indices, norm_vals

    def build_knn_laplacian_sparse(self, context, topk, chunk_size=4096):
        with torch.no_grad():
            N = context.size(0)
            indices, norm_vals = self._get_topk_chunked(context, topk, chunk_size=chunk_size)
            with torch.sparse.check_sparse_tensor_invariants():
                norm_adj_sparse = torch.sparse_coo_tensor(indices, norm_vals, (N, N), device=self.device)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return norm_adj_sparse

    def _build_learned_sparse_adj(self, feats, topk, chunk_size=4096):
        return self._get_topk_chunked(feats, topk, chunk_size=chunk_size)

    def forward(self, build_item_graph=False):
        if build_item_graph:
            weight = self.softmax(self.modal_weight)
            learned_adj = None
            original_adj = None

            if self.v_feat is not None:
                image_feats = F.normalize(self.image_trs(self.image_embedding.weight), p=2, dim=1)
                v_indices, v_vals = self._build_learned_sparse_adj(image_feats, topk=self.knn_k)
                with torch.sparse.check_sparse_tensor_invariants():
                    learned_adj = torch.sparse_coo_tensor(v_indices, v_vals, (self.num_items, self.num_items), device=self.device)
                original_adj = self.image_original_adj

            if self.t_feat is not None:
                text_feats = F.normalize(self.text_trs(self.text_embedding.weight), p=2, dim=1)
                t_indices, t_vals = self._build_learned_sparse_adj(text_feats, topk=self.knn_k)
                with torch.sparse.check_sparse_tensor_invariants():
                    t_adj_sparse = torch.sparse_coo_tensor(t_indices, t_vals, (self.num_items, self.num_items), device=self.device)

                if learned_adj is None:
                    learned_adj = t_adj_sparse
                    original_adj = self.text_original_adj
                else:
                    learned_adj = weight[0] * learned_adj + weight[1] * t_adj_sparse
                    original_adj = weight[0] * original_adj + weight[1] * self.text_original_adj

            if learned_adj is not None:
                self.item_adj = (1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj
            else:
                indices = torch.stack([torch.arange(self.num_items, device=self.device)] * 2, dim=0)
                with torch.sparse.check_sparse_tensor_invariants():
                    self.item_adj = torch.sparse_coo_tensor(indices, torch.ones(self.num_items, device=self.device), (self.num_items, self.num_items))
        else:
            self.item_adj = self.item_adj.detach()  # type: ignore

        h = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.item_adj, h)

        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight), dim=0
        )
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings, [self.num_users, self.num_items], dim=0
        )

        i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)

        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, users, pos_items, neg_items, build_item_graph):
        ua_embeddings, ia_embeddings = self.forward(build_item_graph=build_item_graph)

        u_emb = ua_embeddings[users]
        pos_emb = ia_embeddings[pos_items]
        neg_emb = ia_embeddings[neg_items]

        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)

        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        regularizer = (
            0.5 * (u_emb**2).sum() + 0.5 * (pos_emb**2).sum() + 0.5 * (neg_emb**2).sum()
        )
        regularizer = regularizer / users.size(0)

        return bpr_loss + self.reg_weight * regularizer


class LATTICE(Recommender):
    def __init__(
        self,
        name="LATTICE",
        embedding_size=64,
        feat_embed_dim=64,
        knn_k=10,
        lambda_coeff=0.9,
        n_layers=1,
        reg_weight=1e-4,
        learning_rate=0.0005,
        n_epochs=50,
        batch_size=1024,
        verbose=True,
        device="cpu"
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)
        self.embedding_size = embedding_size
        self.feat_embed_dim = feat_embed_dim
        self.knn_k = knn_k
        self.lambda_coeff = lambda_coeff
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = None
        self.final_u_emb = None
        self.final_i_emb = None

    def _get_ui_adj_mat(self, train_set):
        n_users = train_set.num_users
        n_items = train_set.num_items
        n_nodes = n_users + n_items

        R = train_set.matrix.tocsr()
        top_block = sp.hstack([sp.csr_matrix((n_users, n_users), dtype=np.float32), R])
        bottom_block = sp.hstack([R.T, sp.csr_matrix((n_items, n_items), dtype=np.float32)])
        A = sp.vstack([top_block, bottom_block], format="coo")

        A = (A + sp.eye(n_nodes, dtype=np.float32, format="coo")).tocoo()  # type: ignore

        rowsum = np.array(A.sum(1)).flatten()
        d_inv = np.power(rowsum, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0

        values = d_inv[A.row] * A.data
        indices = torch.LongTensor(np.vstack((A.row, A.col)))

        with torch.sparse.check_sparse_tensor_invariants():
            ret = torch.sparse_coo_tensor(indices, torch.FloatTensor(values), (n_nodes, n_nodes)).to(self.device)
        return ret

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        norm_adj = self._get_ui_adj_mat(train_set)

        v_feat = None
        if train_set.item_image is not None:
            v_feat = torch.tensor(
                train_set.item_image.features[: train_set.num_items],
                dtype=torch.float32,
            ).to(self.device)

        t_feat = None
        if train_set.item_text is not None:
            t_feat = torch.tensor(
                train_set.item_text.features[: train_set.num_items], dtype=torch.float32
            ).to(self.device)

        self.model = RawLATTICE(
            num_users=train_set.num_users,
            num_items=train_set.num_items,
            norm_adj=norm_adj,
            embedding_size=self.embedding_size,
            feat_embed_dim=self.feat_embed_dim,
            knn_k=self.knn_k,
            lambda_coeff=self.lambda_coeff,
            n_layers=self.n_layers,
            reg_weight=self.reg_weight,
            v_feat=v_feat,
            t_feat=t_feat,
            device=self.device  # type: ignore
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs, eta_min=1e-5
        )

        num_ratings = train_set.num_ratings
        users_list, pos_items_list = train_set.matrix.nonzero()


        from tqdm.auto import tqdm

        for epoch in range(self.n_epochs):
            self.model.train()
            indices = np.arange(num_ratings)
            np.random.shuffle(indices)

            build_item_graph_flag = (
                True
            )

            sum_loss = 0.0
            count = 0

            progress_bar = tqdm(
                total=train_set.num_batches(self.batch_size),
                desc=f"Epoch {epoch + 1}/{self.n_epochs}",
                disable=not self.verbose,
            )

            for start_idx in range(0, num_ratings, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                batch_users = users_list[batch_idx]
                batch_pos_items = pos_items_list[batch_idx]

                batch_neg_items = np.random.randint(0, train_set.num_items, size=len(batch_users))

                u_t = torch.tensor(batch_users, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(batch_pos_items, dtype=torch.long, device=self.device)
                neg_t = torch.tensor(batch_neg_items, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                loss = self.model.calculate_loss(
                    u_t, pos_t, neg_t, build_item_graph=build_item_graph_flag
                )

                build_item_graph_flag = False

                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                count += len(batch_users)
                progress_bar.set_postfix(loss=sum_loss / count * 100)
                progress_bar.update(1)
            progress_bar.close()

            scheduler.step()

        self.model.eval()
        with torch.no_grad():
            self.final_u_emb, self.final_i_emb = self.model(build_item_graph=True)
            self.final_u_emb = self.final_u_emb.cpu().numpy()
            self.final_i_emb = self.final_i_emb.cpu().numpy()

    def score(self, user_idx, item_idx=None):
        u_emb = self.final_u_emb[user_idx]  # type: ignore
        if item_idx is None:
            return np.dot(u_emb, self.final_i_emb.T)  # type: ignore
        else:
            return np.dot(u_emb, self.final_i_emb[item_idx].T)  # type: ignore
