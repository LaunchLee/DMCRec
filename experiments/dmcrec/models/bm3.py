import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from cornac.models import Recommender


class RawBM3(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        norm_adj,
        embedding_size,
        n_layers,
        reg_weight,
        cl_weight,
        dropout,
        v_feat=None,
        t_feat=None,
        device="cpu",
    ):
        super(RawBM3, self).__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.embedding_dim = embedding_size
        self.feat_embed_dim = embedding_size
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.cl_weight = cl_weight
        self.dropout = dropout
        self.device = torch.device(device) if isinstance(device, str) else device

        self.norm_adj = norm_adj

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_normal_(self.predictor.weight)

        self.v_feat = v_feat
        self.t_feat = t_feat

        if self.v_feat is not None:
            v_feat_norm = F.normalize(self.v_feat, p=2, dim=1)
            self.image_embedding = nn.Embedding.from_pretrained(
                v_feat_norm.detach(), freeze=True
            )
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)

        if self.t_feat is not None:
            t_feat_norm = F.normalize(self.t_feat, p=2, dim=1)
            self.text_embedding = nn.Embedding.from_pretrained(
                t_feat_norm.detach(), freeze=True
            )
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

    def forward(self):
        h = self.item_id_embedding.weight
        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight), dim=0
        )

        out = ego_embeddings.clone()
        curr_embeddings = ego_embeddings

        for _ in range(self.n_layers):
            curr_embeddings = torch.sparse.mm(self.norm_adj, curr_embeddings)
            out = out + curr_embeddings

        out = out / (self.n_layers + 1.0)
        u_g_embeddings, i_g_embeddings = torch.split(
            out, [self.n_users, self.n_items], dim=0
        )

        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, users, items):
        u_online_ori, i_online_ori = self.forward()

        u_online_b_ori = u_online_ori[users, :]
        i_online_b_ori = i_online_ori[items, :]

        t_online_b, v_online_b = None, None
        t_target_b, v_target_b = None, None
        
        if self.t_feat is not None:
            t_raw_b = self.text_embedding(items)
            t_rep = self.text_trs(t_raw_b)
            t_online_b = self.predictor(t_rep)
            t_target_b = F.dropout(t_rep, self.dropout).detach()

        if self.v_feat is not None:
            v_raw_b = self.image_embedding(items)
            v_rep = self.image_trs(v_raw_b)
            v_online_b = self.predictor(v_rep)
            v_target_b = F.dropout(v_rep, self.dropout).detach()

        with torch.no_grad():
            u_target_b = F.dropout(u_online_b_ori, self.dropout).detach()
            i_target_b = F.dropout(i_online_b_ori, self.dropout).detach()

            t_target_b = F.dropout(t_online_b, self.dropout) if t_online_b is not None else None
            v_target_b = F.dropout(v_online_b, self.dropout) if v_online_b is not None else None

        u_online = self.predictor(u_online_b_ori)
        i_online = self.predictor(i_online_b_ori)

        loss_ui = 1 - cosine_similarity(u_online, i_target_b, dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target_b, dim=-1).mean()

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0

        if t_online_b is not None and t_target_b is not None:
            loss_t = 1 - cosine_similarity(t_online_b, i_target_b, dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_online_b, t_target_b, dim=-1).mean()

        if v_online_b is not None and v_target_b is not None:
            loss_v = 1 - cosine_similarity(v_online_b, i_target_b, dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_online_b, v_target_b, dim=-1).mean()

        u_emb_0 = self.user_embedding.weight[users]
        i_emb_0 = self.item_id_embedding.weight[items]

        reg_loss = (
            0.5 * (u_emb_0 ** 2).sum()
            + 0.5 * (i_emb_0 ** 2).sum()
        ) / users.size(0)

        total_loss = (
            loss_ui + loss_iu
            + self.reg_weight * reg_loss
            + self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt)
        )

        return total_loss


class BM3(Recommender):
    def __init__(
        self,
        name="BM3",
        embedding_size=64,
        n_layers=2,
        dropout=0.3,
        reg_weight=0.01,
        cl_weight=2.0,
        learning_rate=0.001,
        n_epochs=50,
        batch_size=512,
        verbose=True,
        device="cpu"
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.reg_weight = reg_weight
        self.cl_weight = cl_weight
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
        n_nodes = n_users + n_items

        R = train_set.matrix.tocsr()

        top_block = sp.hstack([sp.csr_matrix((n_users, n_users), dtype=np.float32), R])
        bottom_block = sp.hstack([R.T, sp.csr_matrix((n_items, n_items), dtype=np.float32)])
        A = sp.vstack([top_block, bottom_block], format="coo")

        deg = np.array(A.sum(axis=1)).flatten()
        deg[deg == 0] = 1e-12
        d_inv = np.power(deg, -0.5)

        values = d_inv[A.row] * A.data * d_inv[A.col]  # type: ignore

        indices = torch.tensor(np.vstack((A.row, A.col)), dtype=torch.long, device=self.device)  # type: ignore
        data = torch.tensor(values, dtype=torch.float32, device=self.device)

        with torch.sparse.check_sparse_tensor_invariants():
            ret = torch.sparse_coo_tensor(indices, data, (n_nodes, n_nodes)).to(self.device)

        del R, top_block, bottom_block, A, deg, d_inv, values, indices, data
        return ret

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        norm_adj = self._get_norm_adj_mat(train_set)

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

        self.model = RawBM3(
            num_users=train_set.num_users,
            num_items=train_set.num_items,
            norm_adj=norm_adj,
            embedding_size=self.embedding_size,
            n_layers=self.n_layers,
            reg_weight=self.reg_weight,
            cl_weight=self.cl_weight,
            dropout=self.dropout,
            v_feat=v_feat,
            t_feat=t_feat,
            device=self.device    # type: ignore
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

                u_t = torch.tensor(batch_users, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(batch_pos_items, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                loss = self.model.calculate_loss(u_t, pos_t)

                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                sum_loss += loss.item()
                count += len(batch_users)
                progress_bar.set_postfix(loss=sum_loss / count * 100)
                progress_bar.update(1)
            progress_bar.close()

        self.model.eval()
        with torch.no_grad():
            u_emb_raw, i_emb_raw = self.model()
            # u_emb_proj = self.model.predictor(u_emb_raw)

            self.final_u_emb = F.normalize(u_emb_raw, p=2, dim=-1).cpu().numpy()
            self.final_i_emb = F.normalize(i_emb_raw, p=2, dim=-1).cpu().numpy()

    def score(self, user_idx, item_idx=None):
        u_emb = self.final_u_emb[user_idx]  # type: ignore
        if item_idx is None:
            return np.dot(u_emb, self.final_i_emb.T)  # type: ignore
        else:
            return np.dot(u_emb, self.final_i_emb[item_idx].T)  # type: ignore
