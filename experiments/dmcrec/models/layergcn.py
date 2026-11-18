import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cornac.models import Recommender


class RawLayerGCN(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        interaction_matrix,
        embedding_size,
        n_layers,
        dropout,
        device,
    ):
        super(RawLayerGCN, self).__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.n_users = num_users
        self.n_items = num_items
        self.n_nodes = self.n_users + self.n_items

        self.interaction_matrix = interaction_matrix.tocoo().astype(np.float32)

        self.latent_dim = embedding_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.user_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.n_users, self.latent_dim))
        )
        self.item_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.n_items, self.latent_dim))
        )

        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = None
        self.forward_adj = None
        self.pruning_random = False

        self.edge_indices, self.edge_values = self.get_edge_info()

    def _normalize_adj_m(self, indices):
        u_idx = indices[0]
        i_idx = indices[1]

        u_deg = torch.zeros(self.n_users, device=self.device, dtype=torch.float32)
        i_deg = torch.zeros(self.n_items, device=self.device, dtype=torch.float32)
        ones = torch.ones(indices.shape[1], device=self.device, dtype=torch.float32)

        u_deg.scatter_add_(0, u_idx, ones)
        i_deg.scatter_add_(0, i_idx, ones)

        u_deg = u_deg + 1e-7
        i_deg = i_deg + 1e-7

        return torch.pow(u_deg[u_idx], -0.5) * torch.pow(i_deg[i_idx], -0.5)

    def get_edge_info(self):
        with torch.no_grad():
            rows = torch.from_numpy(self.interaction_matrix.row.astype(np.int64)).to(self.device)
            cols = torch.from_numpy(self.interaction_matrix.col.astype(np.int64)).to(self.device)
            edges = torch.stack([rows, cols], dim=0)
            values = self._normalize_adj_m(edges)
            return edges, values

    def get_norm_adj_mat(self):
        inter_M = self.interaction_matrix

        row_forward = inter_M.row
        col_forward = inter_M.col + self.n_users

        row_backward = inter_M.col + self.n_users
        col_backward = inter_M.row

        all_rows = np.concatenate([row_forward, row_backward])
        all_cols = np.concatenate([col_forward, col_backward])
        all_data = np.ones(inter_M.nnz * 2, dtype=np.float32)

        A = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(self.n_nodes, self.n_nodes))

        sumArr = np.array(A.sum(axis=1)).flatten()
        diag = sumArr + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = (D * A * D).tocoo()

        i = torch.LongTensor(np.vstack((L.row, L.col)))
        data = torch.FloatTensor(L.data)
        with torch.sparse.check_sparse_tensor_invariants():
            ret = torch.sparse_coo_tensor(i, data, (self.n_nodes, self.n_nodes))
        return ret

    def pre_epoch_processing(self):
        if self.dropout <= 0.0:
            self.masked_adj = self.norm_adj_matrix
            return

        with torch.no_grad():
            keep_len = int(self.edge_values.size(0) * (1.0 - self.dropout))
            if self.pruning_random:
                keep_idx = torch.tensor(
                    random.sample(range(self.edge_values.size(0)), keep_len),
                    device=self.device,
                    dtype=torch.long
                )
            else:
                keep_idx = torch.multinomial(self.edge_values, keep_len)

            self.pruning_random = True ^ self.pruning_random

            keep_indices = self.edge_indices[:, keep_idx].clone()

            keep_values = self._normalize_adj_m(keep_indices)
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
                    all_indices, all_values, self.norm_adj_matrix.shape
                ).to(self.device)

            del keep_idx, keep_indices, keep_values, all_values, keep_indices_mapped, all_indices

    def get_ego_embeddings(self):
        return torch.cat([self.user_embeddings, self.item_embeddings], 0)

    def forward(self):
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = ego_embeddings
        ui_all_embeddings = torch.zeros_like(ego_embeddings)

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.forward_adj, all_embeddings)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            all_embeddings = torch.einsum("a,ab->ab", _weights, all_embeddings)
            ui_all_embeddings = ui_all_embeddings + all_embeddings

        user_all_embeddings, item_all_embeddings = torch.split(
            ui_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction, reg_weight):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        self.forward_adj = self.masked_adj
        user_all_embeddings, item_all_embeddings = self.forward()

        u_emb = user_all_embeddings[user]
        pos_i_emb = item_all_embeddings[pos_item]
        neg_i_emb = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_emb, pos_i_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_i_emb).sum(dim=1)
        mf_loss = torch.sum(-F.logsigmoid(pos_scores - neg_scores))

        u_ego = self.user_embeddings[user]
        pos_i_ego = self.item_embeddings[pos_item]
        neg_i_ego = self.item_embeddings[neg_item]
        reg_loss = (1 / 2) * (
            u_ego.norm(2).pow(2) + pos_i_ego.norm(2).pow(2) + neg_i_ego.norm(2).pow(2)
        ).sum()

        loss = mf_loss + reg_weight * reg_loss
        return loss

    def full_sort_predict(self, user_indices):
        self.forward_adj = self.norm_adj_matrix
        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user_indices]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


class LayerGCN(Recommender):
    def __init__(
        self,
        name="LayerGCN",
        embedding_size=64,
        n_layers=4,
        reg_weight=1e-03,
        dropout=0.1,
        learning_rate=0.001,
        n_epochs=50,
        batch_size=512,
        verbose=True,
        device="cpu",
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)

        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device

        self.raw_layer_gcn = None

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        self.num_users = train_set.num_users
        self.num_items = train_set.num_items
        csr_matrix = train_set.matrix

        self.raw_layer_gcn = RawLayerGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            interaction_matrix=csr_matrix,
            embedding_size=self.embedding_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
            device=self.device,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.raw_layer_gcn.parameters(), lr=self.learning_rate
        )

        num_ratings = train_set.num_ratings

        from tqdm.auto import tqdm

        for epoch in range(self.n_epochs):
            self.raw_layer_gcn.train()
            sum_loss = 0.0
            count = 0

            progress_bar = tqdm(
                total=train_set.num_batches(self.batch_size),
                desc=f"Epoch {epoch + 1}/{self.n_epochs}",
                disable=not self.verbose,
            )
            self.raw_layer_gcn.pre_epoch_processing()

            users_list, pos_items_list = csr_matrix.nonzero()
            indices = np.arange(num_ratings)
            np.random.shuffle(indices)

            for start_idx in range(0, num_ratings, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                batch_users = users_list[batch_idx]
                batch_pos_items = pos_items_list[batch_idx]

                batch_neg_items = np.random.randint(0, train_set.num_items, size=len(batch_users))

                interaction_tensors = (
                    torch.tensor(batch_users, dtype=torch.long).to(self.device),
                    torch.tensor(batch_pos_items, dtype=torch.long).to(self.device),
                    torch.tensor(batch_neg_items, dtype=torch.long).to(self.device),
                )

                optimizer.zero_grad()
                loss = self.raw_layer_gcn.calculate_loss(
                    interaction_tensors, self.reg_weight
                )
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                count += len(batch_users)
                progress_bar.set_postfix(loss=sum_loss / count * 100)
                progress_bar.update(1)
            progress_bar.close()

        self.raw_layer_gcn.eval()
        with torch.no_grad():
            self.raw_layer_gcn.forward_adj = self.raw_layer_gcn.norm_adj_matrix
            u_e, i_e = self.raw_layer_gcn.forward()
            self.final_u_emb = u_e.cpu().numpy()
            self.final_i_emb = i_e.cpu().numpy()

    def score(self, user_idx, item_idx=None):
        u_emb = self.final_u_emb[user_idx]
        if item_idx is None:
            return np.dot(u_emb, self.final_i_emb.T)
        else:
            return np.dot(u_emb, self.final_i_emb[item_idx].T)
