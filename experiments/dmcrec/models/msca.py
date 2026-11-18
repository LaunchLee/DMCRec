import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cornac.models import Recommender


class RawMSCA(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        norm_R,
        norm_adj,
        struct_original_adj,
        embedding_size,
        n_layers,
        knn_k,
        is_sparse,
        fusion_coeff,
        tau,
        cl_weight,
        reg_weight,
        v_feat=None,
        t_feat=None,
        device="cpu",
    ):
        super(RawMSCA, self).__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.embedding_dim = embedding_size
        self.n_layers = n_layers
        self.knn_k = knn_k
        self.is_sparse = is_sparse
        self.fusion_coeff = fusion_coeff
        self.tau = tau
        self.cl_weight = cl_weight
        self.reg_weight = reg_weight
        self.device = torch.device(device) if isinstance(device, str) else device

        self.norm_R = norm_R
        self.norm_adj = norm_adj
        self.struct_original_adj = struct_original_adj

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.v_feat = v_feat
        self.t_feat = t_feat

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(
                self.v_feat, freeze=True
            )
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
            nn.init.xavier_uniform_(self.image_trs.weight)
            self.image_original_adj = self._build_modal_knn_graph(
                self.image_embedding.weight.detach()
            )

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(
                self.t_feat, freeze=True
            )
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
            nn.init.xavier_uniform_(self.text_trs.weight)
            self.text_original_adj = self._build_modal_knn_graph(
                self.text_embedding.weight.detach()
            )

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=-1)

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid()
        )

    def _build_modal_knn_graph(self, embeds, chunk_size=4096):
        with torch.no_grad():
            norm_embeds = F.normalize(embeds, p=2, dim=1)
            N = self.n_items

            topk_sim_list = []
            indices_list = []

            for i in range(0, N, chunk_size):
                chunk = norm_embeds[i : i + chunk_size]
                sim_chunk = torch.matmul(chunk, norm_embeds.t())
                topk_sim, indices = torch.topk(sim_chunk, k=self.knn_k, dim=-1)

                topk_sim_list.append(topk_sim)
                indices_list.append(indices)

            topk_sim_all = torch.cat(topk_sim_list, dim=0)
            indices_all = torch.cat(indices_list, dim=0)

            row = torch.arange(N, device=self.device).view(-1, 1).expand(-1, self.knn_k).flatten()
            col = indices_all.flatten()
            ii_indices = torch.stack([row, col], dim=0)

            weights = topk_sim_all.flatten() + 1e-7

            row_sum = torch.zeros(N, device=self.device, dtype=torch.float32)
            row_sum.scatter_add_(0, row, weights)
            row_sum = row_sum + 1e-7

            r_inv_sqrt = torch.pow(row_sum, -0.5)
            values = r_inv_sqrt[row] * weights * r_inv_sqrt[col]

            with torch.sparse.check_sparse_tensor_invariants():
                ret = torch.sparse_coo_tensor(ii_indices, values, (N, N), device=self.device)

            del norm_embeds, topk_sim_list, indices_list, topk_sim_all, indices_all, row, col, weights, row_sum, r_inv_sqrt
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return ret

    def semantic_encode(self, view_ii_adj, initial_item_embeds):
        view_item_embeds = torch.sparse.mm(view_ii_adj, initial_item_embeds)
        view_user_embeds = torch.sparse.mm(self.norm_R, view_item_embeds)
        view_embeds = torch.cat([view_user_embeds, view_item_embeds], dim=0)
        return view_embeds

    def forward(self, test=True):
        image_item_embeds, text_item_embeds = None, None
        if self.v_feat is not None:
            image_item_embeds = self.item_id_embedding.weight * self.gate_v(
                self.image_trs(self.image_embedding.weight)
            )
        if self.t_feat is not None:
            text_item_embeds = self.item_id_embedding.weight * self.gate_t(
                self.text_trs(self.text_embedding.weight)
            )

        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_id_embedding.weight], dim=0
        )

        collab_embeds = ego_embeddings.clone()
        curr_embeddings = ego_embeddings
        for _ in range(self.n_layers):
            curr_embeddings = torch.sparse.mm(self.norm_adj, curr_embeddings)
            collab_embeds = collab_embeds + curr_embeddings
        collab_embeds = collab_embeds / (self.n_layers + 1.0)

        struct_embeds = self.semantic_encode(
            self.struct_original_adj, self.item_id_embedding.weight
        )

        image_embeds = (
            self.semantic_encode(self.image_original_adj, image_item_embeds)
            if self.v_feat is not None
            else struct_embeds
        )
        text_embeds = (
            self.semantic_encode(self.text_original_adj, text_item_embeds)
            if self.t_feat is not None
            else struct_embeds
        )

        multi_embed_list = [image_embeds, text_embeds, struct_embeds]
        attn_weights = self.softmax(
            torch.cat([self.query_common(embed) for embed in multi_embed_list], dim=-1)
        )
        redundant_embeds = sum(
            w.unsqueeze(1) * embed
            for w, embed in zip(attn_weights.unbind(dim=1), multi_embed_list)
        )
        multi_embeds = sum(multi_embed_list) - redundant_embeds

        final_embeds = collab_embeds + self.fusion_coeff * multi_embeds
        final_user_embeds, final_item_embeds = torch.split(
            final_embeds, [self.n_users, self.n_items], dim=0
        )

        if test:
            return final_user_embeds, final_item_embeds
        return (
            final_user_embeds,
            final_item_embeds,
            collab_embeds,
            struct_embeds,
            image_embeds,
            text_embeds,
        )

    def cal_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(dim=1)
        neg_scores = (users * neg_items).sum(dim=1)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def cal_cl_loss(self, embed_1, embed_2, tau):
        embed_1 = F.normalize(embed_1, p=2, dim=1)
        embed_2 = F.normalize(embed_2, p=2, dim=1)

        pos = torch.sum(embed_1 * embed_2, dim=-1, keepdim=True)
        tot = torch.matmul(embed_1, embed_2.t())

        gcl_logits = (tot - pos) / tau
        clogits = torch.logsumexp(gcl_logits, dim=1)
        return torch.mean(clogits)

    def cal_reg_loss(self, users, pos_items, neg_items):
        u_e = self.user_embedding(users)
        pos_e = self.item_id_embedding(pos_items)
        neg_e = self.item_id_embedding(neg_items)
        reg_loss = (1 / 2) * (u_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)) / float(len(users))
        return reg_loss

    def calculate_loss(self, users, pos_items, neg_items):
        (
            final_user_embeds,
            final_item_embeds,
            collab_embeds,
            struct_embeds,
            image_embeds,
            text_embeds,
        ) = self.forward(test=False)  # type: ignore

        bpr = self.cal_bpr_loss(
            final_user_embeds[users],
            final_item_embeds[pos_items],
            final_item_embeds[neg_items],
        )
        reg = self.cal_reg_loss(users, pos_items, neg_items)

        collab_user, collab_item = torch.split(
            collab_embeds, [self.n_users, self.n_items], dim=0
        )
        struct_user, struct_item = torch.split(
            struct_embeds, [self.n_users, self.n_items], dim=0
        )
        image_user, image_item = torch.split(
            image_embeds, [self.n_users, self.n_items], dim=0
        )
        text_user, text_item = torch.split(
            text_embeds, [self.n_users, self.n_items], dim=0
        )

        M_u_loss = self.cal_cl_loss(
            collab_user[users], image_user[users], self.tau
        ) + self.cal_cl_loss(collab_user[users], text_user[users], self.tau)
        M_i_loss = self.cal_cl_loss(
            collab_item[pos_items], image_item[pos_items], self.tau
        ) + self.cal_cl_loss(collab_item[pos_items], text_item[pos_items], self.tau)

        C_u_loss = self.cal_cl_loss(collab_user[users], struct_user[users], self.tau)
        C_i_loss = self.cal_cl_loss(
            collab_item[pos_items], struct_item[pos_items], self.tau
        )

        loss = (
            bpr
            + self.cl_weight * (C_u_loss + C_i_loss + M_u_loss + M_i_loss)
            + self.reg_weight * reg
        )
        return loss


class MSCA(Recommender):
    def __init__(
        self,
        name="MSCA",
        embedding_size=64,
        n_layers=3,
        knn_k=10,
        is_sparse=True,
        fusion_coeff=0.3,
        tau=0.2,
        cl_weight=0.005,
        reg_weight=5e-8,
        learning_rate=0.001,
        n_epochs=50,
        batch_size=512,
        verbose=True,
        device="cpu"
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.knn_k = knn_k
        self.is_sparse = is_sparse
        self.fusion_coeff = fusion_coeff
        self.tau = tau
        self.cl_weight = cl_weight
        self.reg_weight = reg_weight
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = None
        self.final_u_emb = None
        self.final_i_emb = None

    def _get_norm_adj_mat(self, train_set):
        n_users = train_set.num_users
        n_items = train_set.num_items

        R_coo = train_set.matrix.tocoo()

        user_degrees = np.array(R_coo.sum(axis=1)).flatten()
        item_degrees = np.array(R_coo.sum(axis=0)).flatten()

        d_user = np.power(user_degrees, -0.5, where=user_degrees > 0, out=np.zeros_like(user_degrees, dtype=float))
        d_item = np.power(item_degrees, -0.5, where=item_degrees > 0, out=np.zeros_like(item_degrees, dtype=float))

        norm_R_data = d_user[R_coo.row] * R_coo.data * d_item[R_coo.col]
        norm_R_mat = sp.coo_matrix((norm_R_data, (R_coo.row, R_coo.col)), shape=(n_users, n_items))

        top_mat = sp.hstack([sp.csr_matrix((n_users, n_users)), norm_R_mat])
        bottom_mat = sp.hstack([norm_R_mat.T, sp.csr_matrix((n_items, n_items))])
        norm_adj_mat = sp.vstack([top_mat, bottom_mat], format="coo")

        norm_R_tensor = torch.sparse_coo_tensor(
            torch.LongTensor(np.vstack((norm_R_mat.row, norm_R_mat.col))),
            torch.FloatTensor(norm_R_mat.data),
            (n_users, n_items)
        ).to(self.device)

        norm_adj_tensor = torch.sparse_coo_tensor(
            torch.LongTensor(np.vstack((norm_adj_mat.row, norm_adj_mat.col))),  # type: ignore
            torch.FloatTensor(norm_adj_mat.data),
            (n_users + n_items, n_users + n_items)
        ).to(self.device)

        del R_coo, user_degrees, item_degrees, d_user, d_item, norm_R_data, norm_R_mat, top_mat, bottom_mat, norm_adj_mat
        return norm_R_tensor, norm_adj_tensor

    def _get_struct_adj_mat(self, train_set):
        n_items = train_set.num_items
        R = train_set.matrix.tocsr()

        cooccurrence = (R.T @ R).tocoo()

        mask = (cooccurrence.row != cooccurrence.col) & (cooccurrence.data > 1)

        filtered_row = cooccurrence.row[mask]
        filtered_col = cooccurrence.col[mask]
        filtered_data = cooccurrence.data[mask]

        if len(filtered_data) > 0:
            edges = torch.tensor(np.vstack((filtered_row, filtered_col)), dtype=torch.long, device=self.device)
            weights = torch.tensor(filtered_data, dtype=torch.float32, device=self.device)

            self_loop = torch.arange(n_items, device=self.device, dtype=torch.long)
            self_loop_edges = torch.stack([self_loop, self_loop], dim=0)
            self_loop_weights = torch.ones(n_items, device=self.device, dtype=torch.float32) * 1e5  # 给自环极大值确保必定被选入 Top-K

            all_edges = torch.cat([edges, self_loop_edges], dim=1)
            all_weights = torch.cat([weights, self_loop_weights], dim=0)

            row_sum = torch.zeros(n_items, device=self.device, dtype=torch.float32)
            row_sum.scatter_add_(0, all_edges[0], torch.ones_like(all_weights))
            row_sum = row_sum + 1e-7

            r_inv_sqrt = torch.pow(row_sum, -0.5)
            norm_values = r_inv_sqrt[all_edges[0]] * r_inv_sqrt[all_edges[1]]

            struct_adj = torch.sparse_coo_tensor(all_edges, norm_values, (n_items, n_items), device=self.device)
        else:
            self_loop = torch.arange(n_items, device=self.device, dtype=torch.long)
            indices = torch.stack([self_loop, self_loop], dim=0)
            struct_adj = torch.sparse_coo_tensor(indices, torch.ones(n_items, device=self.device), (n_items, n_items))

        return struct_adj

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        norm_R, norm_adj = self._get_norm_adj_mat(train_set)
        struct_original_adj = self._get_struct_adj_mat(train_set)

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

        self.model = RawMSCA(
            num_users=train_set.num_users,
            num_items=train_set.num_items,
            norm_R=norm_R,
            norm_adj=norm_adj,
            struct_original_adj=struct_original_adj,
            embedding_size=self.embedding_size,
            n_layers=self.n_layers,
            knn_k=self.knn_k,
            is_sparse=self.is_sparse,
            fusion_coeff=self.fusion_coeff,
            tau=self.tau,
            cl_weight=self.cl_weight,
            reg_weight=self.reg_weight,
            v_feat=v_feat,
            t_feat=t_feat,
            device=self.device  # type: ignore
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        num_ratings = train_set.num_ratings
        users_list, pos_items_list = train_set.matrix.nonzero()

        from tqdm.auto import tqdm

        for epoch in range(self.n_epochs):
            self.model.train()
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

        self.model.eval()
        with torch.no_grad():
            final_user_embeds, final_item_embeds = self.model.forward(test=True)  # type: ignore
            self.final_u_emb = F.normalize(final_user_embeds, p=2, dim=-1).cpu().numpy()
            self.final_i_emb = F.normalize(final_item_embeds, p=2, dim=-1).cpu().numpy()

    def score(self, user_idx, item_idx=None):
        u_emb = self.final_u_emb[user_idx]  # type: ignore
        if item_idx is None:
            return np.dot(u_emb, self.final_i_emb.T)  # type: ignore
        else:
            return np.dot(u_emb, self.final_i_emb[item_idx].T)  # type: ignore
