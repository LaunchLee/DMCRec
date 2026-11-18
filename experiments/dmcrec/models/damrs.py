import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cornac.models import Recommender


class RawDAMRS(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        norm_adj,
        session_adj,
        embedding_size,
        knn_k,
        n_mm_layers,
        n_ui_layers,
        kl_weight,
        neighbor_weight,
        v_feat=None,
        t_feat=None,
        device="cpu",
    ):
        super(RawDAMRS, self).__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.embedding_dim = embedding_size
        self.knn_k = knn_k
        self.n_layers = n_mm_layers
        self.n_ui_layers = n_ui_layers
        self.kl_weight = kl_weight
        self.neighbor_weight = neighbor_weight

        self.device = torch.device(device) if isinstance(device, str) else device

        self.norm_adj = norm_adj
        self.session_adj = session_adj

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
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
            nn.init.xavier_uniform_(self.text_trs.weight)

        self.image_adj, self.text_adj = self._build_knn_adj_matrices()

    def _get_topk_laplacian(self, embeds, chunk_size=4096):
        with torch.no_grad():
            N = embeds.size(0)
            norm_w = F.normalize(embeds, p=2, dim=-1)
            adj_size = (N, N)

            topk_vals_list = []
            topk_ind_list = []

            for i in range(0, N, chunk_size):
                chunk = norm_w[i : i + chunk_size]
                sim_chunk = torch.mm(chunk, norm_w.t())

                mean_val = sim_chunk.mean()
                sim_chunk[sim_chunk < mean_val] = 0.0

                topk_vals, topk_ind = torch.topk(sim_chunk, k=self.knn_k, dim=-1)
                topk_vals_list.append(topk_vals)
                topk_ind_list.append(topk_ind)

                del chunk, sim_chunk

            v_topk_vals = torch.cat(topk_vals_list, dim=0)
            v_knn_ind = torch.cat(topk_ind_list, dim=0)

            row = torch.arange(N, device=self.device).view(-1, 1).expand(-1, self.knn_k).flatten()
            indices = torch.stack((row, v_knn_ind.flatten()), dim=0)

            adj = self._compute_normalized_laplacian(indices, v_topk_vals.flatten() + 1e-7, adj_size)

            del norm_w, topk_vals_list, topk_ind_list, v_topk_vals, v_knn_ind, row, indices
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return adj

    def _build_knn_adj_matrices(self):
        adj_size = (self.n_items, self.n_items)

        if self.v_feat is None and self.t_feat is None:
            indices = torch.stack([torch.arange(self.n_items, device=self.device)] * 2, dim=0)
            with torch.sparse.check_sparse_tensor_invariants():
                empty_adj = torch.sparse_coo_tensor(indices, torch.ones(self.n_items, device=self.device), adj_size)
            return empty_adj, empty_adj

        v_w = self.image_embedding.weight.detach() if self.v_feat is not None else torch.randn(self.n_items, self.embedding_dim, device=self.device)
        t_w = self.text_embedding.weight.detach() if self.t_feat is not None else torch.randn(self.n_items, self.embedding_dim, device=self.device)

        v_adj = self._get_topk_laplacian(v_w)
        t_adj = self._get_topk_laplacian(t_w)

        return v_adj, t_adj

    def _compute_normalized_laplacian(self, indices, weights, adj_size):
        row = indices[0]
        col = indices[1]

        row_sum = torch.zeros(adj_size[0], device=self.device, dtype=torch.float32)
        row_sum.scatter_add_(0, row, weights)
        row_sum = row_sum + 1e-7

        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[row] * weights * r_inv_sqrt[col]

        with torch.sparse.check_sparse_tensor_invariants():
            ret = torch.sparse_coo_tensor(indices, values, adj_size)
        return ret

    def forward(self):
        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight), dim=0
        )
        out = ego_embeddings.clone()
        curr_embeddings = ego_embeddings

        for _ in range(self.n_ui_layers):
            curr_embeddings = torch.sparse.mm(self.norm_adj, curr_embeddings)
            out = out + curr_embeddings

        out = out / (self.n_ui_layers + 1.0)
        u_g_embeddings, i_g_embeddings = torch.split(
            out, [self.n_users, self.n_items], dim=0
        )

        h_t = self.item_id_embedding.weight.clone()
        h_v = self.item_id_embedding.weight.clone()
        h_s = self.item_id_embedding.weight.clone()

        for _ in range(self.n_layers):
            h_t = torch.sparse.mm(self.text_adj, h_t)
            h_v = torch.sparse.mm(self.image_adj, h_v)
            h_s = torch.sparse.mm(self.session_adj, h_s)

        return u_g_embeddings, i_g_embeddings, h_t, h_v, h_s

    def label_prediction(self, emb_selected, all_emb):
        n_emb = F.normalize(emb_selected, dim=1)
        n_all_emb = F.normalize(all_emb, dim=1)
        prob = torch.mm(n_emb, n_all_emb.t())
        return F.softmax(prob, dim=1)

    def generate_pesudo_labels(self, prob1, prob2, prob3):
        positive = prob1 + prob2 + prob3 + prob3
        _, mm_pos_ind = torch.topk(positive, 10, dim=-1)
        prob = prob3.clone()
        prob.scatter_(1, mm_pos_ind, 0)
        _, single_pos_ind = torch.topk(prob, 10, dim=-1)
        return mm_pos_ind, single_pos_ind

    def neighbor_discrimination(
        self, mm_positive, s_positive, emb_selected, sub_all_emb, temperature=0.2
    ):
        n_all_emb = F.normalize(sub_all_emb, dim=1)
        n_emb = F.normalize(emb_selected, dim=1)

        mm_pos_emb = n_all_emb[mm_positive]
        s_pos_emb = n_all_emb[s_positive]

        emb2 = n_emb.unsqueeze(1).expand(-1, 10, -1)

        mm_pos_score = (emb2 * mm_pos_emb).sum(dim=2)
        s_pos_score = (emb2 * s_pos_emb).sum(dim=2)
        ttl_score = torch.matmul(n_emb, n_all_emb.t())

        mm_pos_score = torch.exp(mm_pos_score / temperature).sum(dim=1)
        s_pos_score = torch.exp(s_pos_score / temperature).sum(dim=1)
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

        cl_loss = -torch.log(mm_pos_score / (ttl_score) + 1e-10) - torch.log(
            s_pos_score / (ttl_score - mm_pos_score) + 1e-10
        )
        return torch.mean(cl_loss)

    def KL_loss_calc(self, p1, p2):
        return (
            p1 * torch.log(p1 + 1e-10)
            - p1 * torch.log(p2 + 1e-10)
            + (1 - p1) * torch.log(1 - p1 + 1e-10)
            - (1 - p1) * torch.log(1 - p2 + 1e-10)
        )

    def get_weight_modal(
        self, users, pos_items, neg_items, user_embeddings, h_t, h_v, h_s
    ):
        u_g = user_embeddings[users]

        p_t = (u_g * F.normalize(h_t[pos_items], dim=-1)).sum(dim=1)
        p_v = (u_g * F.normalize(h_s[pos_items], dim=-1)).sum(dim=1)
        p_s = (u_g * F.normalize(h_v[pos_items], dim=-1)).sum(dim=1)

        n_t = (u_g * F.normalize(h_t[neg_items], dim=-1)).sum(dim=1)
        n_v = (u_g * F.normalize(h_s[neg_items], dim=-1)).sum(dim=1)
        n_s = (u_g * F.normalize(h_v[neg_items], dim=-1)).sum(dim=1)

        p_tensor = torch.sigmoid(torch.stack([p_t, p_v, p_s]))
        p_variance = torch.var(p_tensor, dim=0).detach()
        p_mean_value = torch.mean(p_tensor, dim=0).detach()
        p_max_value, _ = torch.max(p_tensor, dim=0)

        n_tensor = torch.sigmoid(torch.stack([n_t, n_v, n_s]))
        n_mean_value = torch.mean(n_tensor).detach()

        pos_weight = torch.clamp(
            p_mean_value * torch.pow(torch.exp(-p_variance), 2.0), 0, 1
        ).detach()

        mask = (p_mean_value < n_mean_value).float()
        neg_weight = torch.clamp((p_max_value - n_mean_value) * mask, 0, 1).detach()

        return pos_weight, neg_weight

    def bpr_loss(self, users, pos_items, neg_items, p_weight, n_weight):
        pos_scores = (users * pos_items).sum(dim=1)
        neg_scores = (users * neg_items).sum(dim=1)
        p_maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10) * p_weight
        n_maxi = torch.log(torch.sigmoid(neg_scores - pos_scores) + 1e-10) * n_weight
        return -torch.mean(p_maxi + n_maxi)

    def label_prediction_batch(self, emb_selected, sub_all_emb):
        n_emb = F.normalize(emb_selected, dim=1)
        n_all_emb = F.normalize(sub_all_emb, dim=1)
        prob = torch.mm(n_emb, n_all_emb.t())
        return F.softmax(prob, dim=1)

    def calculate_loss(self, users, pos_items, neg_items):
        user_embeddings, item_embeddings, h_t, h_v, h_s = self.forward()

        u_id = torch.unique(users, sorted=False)
        i_id = torch.unique(torch.cat((pos_items, neg_items)), sorted=False)

        with torch.no_grad():
            lp_t = self.label_prediction_batch(h_t[i_id], h_t[i_id])
            lp_v = self.label_prediction_batch(h_v[i_id], h_v[i_id])
            lp_s = self.label_prediction_batch(h_s[i_id], h_s[i_id])

            mm_p_s, s_p_s = self.generate_pesudo_labels(lp_t, lp_v, lp_s)
            mm_p_v, s_p_v = self.generate_pesudo_labels(lp_t, lp_s, lp_v)
            mm_p_t, s_p_t = self.generate_pesudo_labels(lp_v, lp_s, lp_t)

        nd_loss_1 = self.neighbor_discrimination(mm_p_s, s_p_s, h_s[i_id], h_s[i_id])
        nd_loss_2 = self.neighbor_discrimination(mm_p_v, s_p_v, h_v[i_id], h_v[i_id])
        nd_loss_3 = self.neighbor_discrimination(mm_p_t, s_p_t, h_t[i_id], h_t[i_id])

        neighbor_dis_loss = (nd_loss_1 + nd_loss_2 + nd_loss_3) / 3.0

        it_embeddings = (h_t + h_s + h_v) / 3.0
        p_g_pos = torch.sigmoid((user_embeddings[users] * F.normalize(item_embeddings[pos_items], dim=-1)).sum(dim=-1))
        p_t_pos = torch.sigmoid((user_embeddings[users] * F.normalize(it_embeddings[pos_items], dim=-1)).sum(dim=-1))

        KL_loss = torch.mean(self.KL_loss_calc(p_g_pos, p_t_pos) + self.KL_loss_calc(p_t_pos, p_g_pos))

        p_weight, n_weight = self.get_weight_modal(
            users, pos_items, neg_items, user_embeddings, h_t, h_v, h_s
        )

        ia_embeddings = item_embeddings + (h_t + h_v + h_s) / 3.0
        batch_mf_loss = self.bpr_loss(
            user_embeddings[users],
            ia_embeddings[pos_items],
            ia_embeddings[neg_items],
            p_weight,
            n_weight,
        )

        return (
            batch_mf_loss
            + self.neighbor_weight * neighbor_dis_loss
            + KL_loss * self.kl_weight
        )


class DAMRS(Recommender):
    def __init__(
        self,
        name="DA-MRS",
        embedding_size=64,
        knn_k=10,
        n_mm_layers=1,
        n_ui_layers=2,
        kl_weight=1.0,
        neighbor_weight=0.001,
        learning_rate=0.001,
        n_epochs=15,
        batch_size=1024,
        verbose=True,
        device="cpu"
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)
        self.embedding_size = embedding_size
        self.knn_k = knn_k
        self.n_mm_layers = n_mm_layers
        self.n_ui_layers = n_ui_layers
        self.kl_weight = kl_weight
        self.neighbor_weight = neighbor_weight
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.final_u_emb = None
        self.final_i_emb = None

    def _get_norm_adj_mat(self, train_set):
        n_users = train_set.num_users
        n_items = train_set.num_items
        n_nodes = n_users + n_items

        R = train_set.matrix.tocsr()
        top_block = sp.hstack([sp.csr_matrix((n_users, n_users), dtype=np.float32), R])
        bottom_block = sp.hstack([R.T, sp.csr_matrix((n_items, n_items), dtype=np.float32)])
        A = sp.vstack([top_block, bottom_block], format="coo")

        eye = sp.eye(n_nodes, dtype=np.float32, format="coo")  # type: ignore
        A = (A + eye).tocoo()

        deg = np.array(A.sum(axis=1)).flatten()
        deg[deg == 0] = 1e-12
        d_inv = np.power(deg, -0.5)
        values = d_inv[A.row] * A.data * d_inv[A.col]

        indices = torch.from_numpy(np.vstack((A.row, A.col)).astype(np.int64))
        data = torch.from_numpy(values.astype(np.float32))

        with torch.sparse.check_sparse_tensor_invariants():
            ret = torch.sparse_coo_tensor(indices, data, (n_nodes, n_nodes)).to(self.device)

        del R, top_block, bottom_block, A, eye, deg, d_inv, values, indices, data
        return ret

    def _build_native_session_adj(self, train_set):
        n_items = train_set.num_items
        adj_size = (n_items, n_items)

        R = train_set.matrix.tocsr()
        co_occurrence = R.T.dot(R).tocoo()

        edges_x = torch.from_numpy(co_occurrence.row.astype(np.int64)).to(self.device)
        edges_y = torch.from_numpy(co_occurrence.col.astype(np.int64)).to(self.device)
        weights = torch.from_numpy(co_occurrence.data.astype(np.float32)).to(self.device)
        indices = torch.stack((edges_x, edges_y), 0)

        row_sum = torch.zeros(n_items, device=self.device, dtype=torch.float32)
        row_sum.scatter_add_(0, edges_x, weights)
        row_sum = row_sum + 1e-7

        r_inv_sqrt = torch.pow(row_sum, -0.5)
        norm_values = r_inv_sqrt[edges_x] * weights * r_inv_sqrt[edges_y]

        with torch.sparse.check_sparse_tensor_invariants():
            ret = torch.sparse_coo_tensor(indices, norm_values, adj_size)

        del co_occurrence, edges_x, edges_y, weights, indices, row_sum, r_inv_sqrt
        return ret

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        norm_adj = self._get_norm_adj_mat(train_set)
        session_adj = self._build_native_session_adj(train_set)

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

        self.model = RawDAMRS(
            num_users=train_set.num_users,
            num_items=train_set.num_items,
            norm_adj=norm_adj,
            session_adj=session_adj,
            embedding_size=self.embedding_size,
            knn_k=self.knn_k,
            n_mm_layers=self.n_mm_layers,
            n_ui_layers=self.n_ui_layers,
            kl_weight=self.kl_weight,
            neighbor_weight=self.neighbor_weight,
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
            user_embeddings, item_embeddings, h_t, h_v, h_s = self.model.forward()
            i_embedding = (h_v + h_t + h_s) / 3.0
            all_item_e = item_embeddings + i_embedding

            self.final_u_emb = F.normalize(user_embeddings, p=2, dim=-1).cpu().numpy()
            self.final_i_emb = F.normalize(all_item_e, p=2, dim=-1).cpu().numpy()

    def score(self, user_idx, item_idx=None):
        u_emb = self.final_u_emb[user_idx]  # type: ignore
        if item_idx is None:
            return np.dot(u_emb, self.final_i_emb.T)  # type: ignore
        else:
            return np.dot(u_emb, self.final_i_emb[item_idx].T)  # type: ignore
