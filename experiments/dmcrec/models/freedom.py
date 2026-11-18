import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cornac.models import Recommender


class RawFREEDOM(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        norm_adj,
        edge_indices,
        edge_values,
        embedding_size,
        feat_embed_dim,
        knn_k,
        lambda_coeff,
        n_mm_layers,
        n_ui_layers,
        reg_weight,
        mm_image_weight,
        dropout,
        v_feat=None,
        t_feat=None,
        device="cpu",
    ):
        super(RawFREEDOM, self).__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.embedding_dim = embedding_size
        self.feat_embed_dim = feat_embed_dim
        self.knn_k = knn_k
        self.lambda_coeff = lambda_coeff
        self.n_layers = n_mm_layers
        self.n_ui_layers = n_ui_layers
        self.reg_weight = reg_weight
        self.mm_image_weight = mm_image_weight
        self.dropout = dropout
        self.device = device

        self.norm_adj = norm_adj
        self.edge_indices = edge_indices
        self.edge_values = edge_values
        self.masked_adj = None

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.v_feat = v_feat
        self.t_feat = t_feat
        self.mm_adj = None

        if self.v_feat is not None:
            v_feat_norm = F.normalize(self.v_feat, p=2, dim=1)
            self.image_embedding = nn.Embedding.from_pretrained(
                v_feat_norm, freeze=True
            )
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.image_trs.weight)

        if self.t_feat is not None:
            t_feat_norm = F.normalize(self.t_feat, p=2, dim=1)
            self.text_embedding = nn.Embedding.from_pretrained(
                t_feat_norm, freeze=True
            )
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.text_trs.weight)

        self._build_and_freeze_mm_graph()

    def _get_topk_indices_chunked(self, embeds, chunk_size=4096):
        with torch.no_grad():
            N = embeds.size(0)
            context_norm = F.normalize(embeds, p=2, dim=-1)
            knn_ind_list = []

            for i in range(0, N, chunk_size):
                chunk = context_norm[i : i + chunk_size]
                sim_chunk = torch.mm(chunk, context_norm.t())
                _, knn_ind = torch.topk(sim_chunk, k=self.knn_k, dim=-1)
                knn_ind_list.append(knn_ind)
                del sim_chunk, chunk
            all_knn_ind = torch.cat(knn_ind_list, dim=0)

            row = torch.arange(N, device=self.device).view(-1, 1).expand(-1, self.knn_k).flatten()
            col = all_knn_ind.flatten()
            indices = torch.stack((row, col), dim=0)

            del context_norm, knn_ind_list, all_knn_ind, row, col
            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                torch.cuda.empty_cache()
            return indices

    def _build_and_freeze_mm_graph(self, chunk_size=4096):
        with torch.no_grad():
            image_adj = None
            text_adj = None
            adj_size = (self.n_items, self.n_items)

            if self.v_feat is not None:
                img_w = self.image_embedding.weight.detach()
                indices_v = self._get_topk_indices_chunked(img_w, chunk_size=chunk_size)
                image_adj = self._compute_sparse_normalized_laplacian(indices_v, adj_size)

            if self.t_feat is not None:
                txt_w = self.text_embedding.weight.detach()
                indices_t = self._get_topk_indices_chunked(txt_w, chunk_size=chunk_size)
                text_adj = self._compute_sparse_normalized_laplacian(indices_t, adj_size)

            if image_adj is not None and text_adj is not None:
                self.mm_adj = (self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj)
            elif image_adj is not None:
                self.mm_adj = image_adj
            elif text_adj is not None:
                self.mm_adj = text_adj
            else:
                indices = torch.stack([torch.arange(self.n_items, device=self.device)] * 2, dim=0)
                with torch.sparse.check_sparse_tensor_invariants():
                    self.mm_adj = torch.sparse_coo_tensor(indices, torch.ones(self.n_items, device=self.device), adj_size)

    def _compute_sparse_normalized_laplacian(self, indices, adj_size):
        row, col = indices[0], indices[1]

        row_sum = torch.zeros(adj_size[0], device=self.device, dtype=torch.float32)
        row_sum.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        row_sum = row_sum + 1e-7

        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[row] * r_inv_sqrt[col]

        with torch.sparse.check_sparse_tensor_invariants():
            ret = torch.sparse_coo_tensor(indices, values, adj_size)
        return ret

    def pre_epoch_processing(self):
        if self.dropout <= 0.0:
            self.masked_adj = self.norm_adj
            return

        with torch.no_grad():
            degree_len = int(self.edge_values.size(0) * (1.0 - self.dropout))

            degree_idx = torch.multinomial(self.edge_values.to(self.device), degree_len, replacement=False)
            keep_indices = self.edge_indices.to(self.device)[:, degree_idx]

            u_deg = torch.zeros(self.n_users, device=self.device, dtype=torch.float32)
            i_deg = torch.zeros(self.n_items, device=self.device, dtype=torch.float32)
            ones = torch.ones(keep_indices.shape[1], device=self.device, dtype=torch.float32)

            u_deg.scatter_add_(0, keep_indices[0], ones)
            i_deg.scatter_add_(0, keep_indices[1], ones)

            u_deg = u_deg + 1e-7
            i_deg = i_deg + 1e-7

            keep_values = torch.pow(u_deg[keep_indices[0]], -0.5) * torch.pow(i_deg[keep_indices[1]], -0.5)
            all_values = torch.cat((keep_values, keep_values))

            keep_indices_mapped = keep_indices.clone()
            keep_indices_mapped[1] += self.n_users

            all_indices = torch.cat(
                (
                    keep_indices_mapped,
                    torch.stack((keep_indices_mapped[1], keep_indices_mapped[0]), dim=0),
                ),
                dim=1,
            )

            with torch.sparse.check_sparse_tensor_invariants():
                self.masked_adj = torch.sparse_coo_tensor(
                    all_indices, all_values, self.norm_adj.shape
                )

            del degree_idx, keep_indices, u_deg, i_deg, ones, keep_values, all_values, keep_indices_mapped, all_indices

    def forward(self, adj):
        h = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight), dim=0
        )
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )

        return u_g_embeddings, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(dim=1)
        neg_scores = (users * neg_items).sum(dim=1)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def calculate_loss(self, users, pos_items, neg_items):
        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)

        u_g = ua_embeddings[users]
        pos_g = ia_embeddings[pos_items]
        neg_g = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g, pos_g, neg_g)

        mf_v_loss, mf_t_loss = 0.0, 0.0

        if self.t_feat is not None:
            t_pos_raw = self.text_embedding(pos_items)
            t_neg_raw = self.text_embedding(neg_items)
            text_pos = F.normalize(self.text_trs(t_pos_raw), p=2, dim=1)
            text_neg = F.normalize(self.text_trs(t_neg_raw), p=2, dim=1)
            mf_t_loss = self.bpr_loss(u_g, text_pos, text_neg)

        if self.v_feat is not None:
            v_pos_raw = self.image_embedding(pos_items)
            v_neg_raw = self.image_embedding(neg_items)
            img_pos = F.normalize(self.image_trs(v_pos_raw), p=2, dim=1)
            img_neg = F.normalize(self.image_trs(v_neg_raw), p=2, dim=1)
            mf_v_loss = self.bpr_loss(u_g, img_pos, img_neg)

        total_loss = batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)
        return total_loss


class FREEDOM(Recommender):
    def __init__(
        self,
        name="FREEDOM",
        embedding_size=64,
        feat_embed_dim=64,
        knn_k=10,
        lambda_coeff=0.9,
        n_mm_layers=1,
        n_ui_layers=2,
        reg_weight=1e-4,
        mm_image_weight=0.1,
        dropout=0.8,
        learning_rate=0.0005,
        n_epochs=50,
        batch_size=512,
        verbose=True,
        device="cpu"
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)
        self.embedding_size = embedding_size
        self.feat_embed_dim = feat_embed_dim
        self.knn_k = knn_k
        self.lambda_coeff = lambda_coeff
        self.n_mm_layers = n_mm_layers
        self.n_ui_layers = n_ui_layers
        self.reg_weight = reg_weight
        self.mm_image_weight = mm_image_weight
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = None
        self.final_u_emb = None
        self.final_i_emb = None

    def _get_base_graph_info(self, train_set):
        n_users = train_set.num_users
        n_items = train_set.num_items
        n_nodes = n_users + n_items

        R = train_set.matrix.tocsr().tocoo()
        edges = torch.tensor(np.vstack((R.row, R.col)), dtype=torch.long, device=self.device)

        u_deg = torch.zeros(n_users, device=self.device, dtype=torch.float32)
        i_deg = torch.zeros(n_items, device=self.device, dtype=torch.float32)

        ones = torch.ones(edges.shape[1], device=self.device, dtype=torch.float32)
        u_deg.scatter_add_(0, edges[0], ones)
        i_deg.scatter_add_(0, edges[1], ones)

        u_deg = u_deg + 1e-7
        i_deg = i_deg + 1e-7

        # D_u^(-1/2) * 1.0 * D_i^(-1/2)
        edge_values = torch.pow(u_deg[edges[0]], -0.5) * torch.pow(i_deg[edges[1]], -0.5)

        R_csr = train_set.matrix.tocsr()
        top_block = sp.hstack([sp.csr_matrix((n_users, n_users), dtype=np.float32), R_csr])
        bottom_block = sp.hstack([R_csr.T, sp.csr_matrix((n_items, n_items), dtype=np.float32)])
        A = sp.vstack([top_block, bottom_block], format="coo")

        A = (A + sp.eye(n_nodes, dtype=np.float32, format="coo")).tocoo()  # type: ignore

        deg_all = np.array(A.sum(axis=1)).flatten() + 1e-7
        d_inv = np.power(deg_all, -0.5)
        norm_values = d_inv[A.row] * A.data * d_inv[A.col]

        indices = torch.tensor(np.vstack((A.row, A.col)), dtype=torch.long)
        data = torch.tensor(norm_values, dtype=torch.float32)

        with torch.sparse.check_sparse_tensor_invariants():
            norm_adj = torch.sparse_coo_tensor(indices, data, (n_nodes, n_nodes)).to(self.device)

        del R, R_csr, top_block, bottom_block, A, deg_all, d_inv, norm_values, indices, data, ones
        return norm_adj, edges.cpu(), edge_values

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        norm_adj, edge_indices, edge_values = self._get_base_graph_info(train_set)

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

        self.model = RawFREEDOM(
            num_users=train_set.num_users,
            num_items=train_set.num_items,
            norm_adj=norm_adj,
            edge_indices=edge_indices,
            edge_values=edge_values,
            embedding_size=self.embedding_size,
            feat_embed_dim=self.feat_embed_dim,
            knn_k=self.knn_k,
            lambda_coeff=self.lambda_coeff,
            n_mm_layers=self.n_mm_layers,
            n_ui_layers=self.n_ui_layers,
            reg_weight=self.reg_weight,
            mm_image_weight=self.mm_image_weight,
            dropout=self.dropout,
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
            self.model.pre_epoch_processing()

            indices = np.arange(num_ratings)
            np.random.shuffle(indices)

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
                loss = self.model.calculate_loss(u_t, pos_t, neg_t)
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
            u_emb, i_emb = self.model(self.model.norm_adj)
            self.final_u_emb = F.normalize(u_emb, p=2, dim=-1).cpu().numpy()
            self.final_i_emb = F.normalize(i_emb, p=2, dim=-1).cpu().numpy()

    def score(self, user_idx, item_idx=None):
        u_emb = self.final_u_emb[user_idx]  # type: ignore
        if item_idx is None:
            return np.dot(u_emb, self.final_i_emb.T)  # type: ignore
        else:
            return np.dot(u_emb, self.final_i_emb[item_idx].T)  # type: ignore
