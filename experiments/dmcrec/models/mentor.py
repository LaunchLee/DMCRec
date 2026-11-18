import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from cornac.models import Recommender


class MENTORBaseGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add", **kwargs):
        super(MENTORBaseGCN, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == "add":
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out


class MENTORGCNModule(torch.nn.Module):
    def __init__(
        self,
        num_user,
        num_item,
        dim_x,
        aggr_mode,
        dim_latent=64,
        device=None,
        features=None,
    ):
        super(MENTORGCNModule, self).__init__()
        self.device = device
        self.num_item = num_item
        self.dim_latent = dim_latent
        self.dim_feat = features.size(1) if features is not None else dim_x

        self.preference = nn.Parameter(torch.empty(num_user, self.dim_latent))
        nn.init.xavier_normal_(self.preference)

        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

        self.conv_embed_1 = MENTORBaseGCN(
            self.dim_latent, self.dim_latent, aggr=aggr_mode
        )

    def forward(self, edge_index, features, perturbed=False):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features)))

        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x, p=2, dim=-1)

        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h)
            h = h + torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1

        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h_1)
            h_1 = h_1 + torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1

        x_hat = x + h + h_1
        return x_hat, self.preference


class RawMENTORNet(nn.Module):
    def __init__(
        self, config, num_users, num_items, v_feat, t_feat, raw_edge_index, device
    ):
        super(RawMENTORNet, self).__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.device = torch.device(device) if isinstance(device, str) else device

        dim_x = config["embedding_size"]
        # self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.knn_k = config["knn_k"]
        self.mm_image_weight = config["mm_image_weight"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        self.align_weight = config["align_weight"]
        self.mask_weight_g = config["mask_weight_g"]
        self.mask_weight_f = config["mask_weight_f"]
        self.temp = config["temp"]

        self.v_feat = v_feat.detach()
        self.t_feat = t_feat.detach()
        self.dim_latent = dim_x
        self.mlp = nn.Linear(2 * dim_x, 2 * dim_x)

        with torch.no_grad():
            self.edge_index = torch.cat((raw_edge_index, raw_edge_index[[1, 0]]), dim=1).to(self.device)

            drop_item = torch.randperm(self.n_items, device=self.device)[: int(self.n_items * 0.1)]
            self.dropv_node_idx = drop_item[: int(len(drop_item) * 1 / 3)]
            self.dropt_node_idx = drop_item[int(len(drop_item) * 2 / 3) :]

            item_in_edges = raw_edge_index[1] - self.n_users

            mask_v = ~torch.isin(item_in_edges, self.dropv_node_idx)
            mask_t = ~torch.isin(item_in_edges, self.dropt_node_idx)

            raw_dropv = raw_edge_index[:, mask_v]
            raw_dropt = raw_edge_index[:, mask_t]

            self.edge_index_dropv = torch.cat((raw_dropv, raw_dropv[[1, 0]]), dim=1).to(self.device)
            self.edge_index_dropt = torch.cat((raw_dropt, raw_dropt[[1, 0]]), dim=1).to(self.device)

        self.v_gcn = MENTORGCNModule(
            num_users, num_items, dim_x, "add", device=self.device, features=self.v_feat, dim_latent=self.dim_latent
        )
        self.v_gcn_n1 = MENTORGCNModule(
            num_users, num_items, dim_x, "add", device=self.device, features=self.v_feat, dim_latent=self.dim_latent
        )
        self.v_gcn_n2 = MENTORGCNModule(
            num_users, num_items, dim_x, "add", device=self.device, features=self.v_feat, dim_latent=self.dim_latent
        )

        self.t_gcn = MENTORGCNModule(
            num_users, num_items, dim_x, "add", device=self.device, features=self.t_feat, dim_latent=self.dim_latent
        )
        self.t_gcn_n1 = MENTORGCNModule(
            num_users, num_items, dim_x, "add", device=self.device, features=self.t_feat, dim_latent=self.dim_latent
        )
        self.t_gcn_n2 = MENTORGCNModule(
            num_users, num_items, dim_x, "add", device=self.device, features=self.t_feat, dim_latent=self.dim_latent
        )

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(
                torch.tensor(
                    np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32
                )
            ).to(self.device)
        )
        self.id_gcn = MENTORGCNModule(
            num_users,
            num_items,
            dim_x,
            "add",
            device=self.device,
            features=self.id_feat,
        )

        self.weight_u = nn.Parameter(
            nn.init.xavier_normal_(
                torch.tensor(np.random.randn(self.n_users, 2, 1), dtype=torch.float32)
            ).to(self.device)
        )
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
        _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
        _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
        self.mm_adj = (
            self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        )

    def get_knn_adj_mat(self, embs, chunk_size=4096):
        with torch.no_grad():
            num_nodes = embs.shape[0]
            norm = F.normalize(embs, p=2, dim=-1)

            topk_sim_list, knn_ind_list = [], []

            for i in range(0, num_nodes, chunk_size):
                chunk = norm[i : i + chunk_size]
                sim_chunk = torch.mm(chunk, norm.t())

                topk_sim, knn_ind = torch.topk(sim_chunk, k=self.knn_k, dim=-1)
                topk_sim_list.append(topk_sim)
                knn_ind_list.append(knn_ind)

                del chunk, sim_chunk

            topk_sim_all = torch.cat(topk_sim_list, dim=0)
            knn_ind_all = torch.cat(knn_ind_list, dim=0)

            row = torch.arange(num_nodes, device=self.device).view(-1, 1).expand(-1, self.knn_k).flatten()
            col = knn_ind_all.flatten()
            indices = torch.stack((row, col), dim=0)

            weights = topk_sim_all.flatten() + 1e-7

            row_sum = torch.zeros(num_nodes, device=self.device, dtype=torch.float32)
            row_sum.scatter_add_(0, row, weights)
            row_sum = row_sum + 1e-7

            r_inv_sqrt = torch.pow(row_sum, -0.5)
            values = r_inv_sqrt[row] * weights * r_inv_sqrt[col]

            with torch.sparse.check_sparse_tensor_invariants():
                adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=self.device)

            del norm, topk_sim_list, knn_ind_list, topk_sim_all, knn_ind_all, row, col, weights, row_sum, r_inv_sqrt
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return indices, adj

    def buildItemGraph(self, h):
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = torch.exp((view1 * view2).sum(dim=-1) / temp)
        ttl_score = torch.exp(torch.matmul(view1, view2.t()) / temp).sum(dim=1)
        return torch.mean(-torch.log(pos_score / (ttl_score + 1e-7)))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = (
            interaction[0],
            interaction[1],
            interaction[2],
        )
        pos_item_nodes = pos_item_nodes + self.n_users
        neg_item_nodes = neg_item_nodes + self.n_users

        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, self.id_preference = self.id_gcn(
            self.edge_index, self.id_feat
        )

        self.v_rep_n1, _ = self.v_gcn_n1(
            self.edge_index, self.v_feat, perturbed=True
        )
        self.t_rep_n1, _ = self.t_gcn_n1(
            self.edge_index, self.t_feat, perturbed=True
        )
        self.v_rep_n2, _ = self.v_gcn_n2(
            self.edge_index, self.v_feat, perturbed=True
        )
        self.t_rep_n2, _ = self.t_gcn_n2(
            self.edge_index, self.t_feat, perturbed=True
        )

        representation = torch.cat((self.v_rep, self.t_rep), dim=1)
        guide_representation = torch.cat((self.id_rep, self.id_rep), dim=1)
        v_representation = torch.cat((self.v_rep, self.v_rep), dim=1)
        t_representation = torch.cat((self.t_rep, self.t_rep), dim=1)
        representation_n1 = torch.cat((self.v_rep_n1, self.t_rep_n1), dim=1)
        representation_n2 = torch.cat((self.v_rep_n2, self.t_rep_n2), dim=1)

        user_rep = torch.cat(
            (
                torch.unsqueeze(self.v_rep, 2)[: self.n_users],
                torch.unsqueeze(self.t_rep, 2)[: self.n_users],
            ),
            dim=2,
        )
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

        guide_user_rep = torch.cat(
            (
                torch.unsqueeze(self.id_rep, 2)[: self.n_users],
                torch.unsqueeze(self.id_rep, 2)[: self.n_users],
            ),
            dim=2,
        )
        guide_user_rep = torch.cat(
            (guide_user_rep[:, :, 0], guide_user_rep[:, :, 1]), dim=1
        )

        v_user_rep = torch.cat(
            (
                torch.unsqueeze(self.v_rep, 2)[: self.n_users],
                torch.unsqueeze(self.v_rep, 2)[: self.n_users],
            ),
            dim=2,
        )
        v_user_rep = torch.cat((v_user_rep[:, :, 0], v_user_rep[:, :, 1]), dim=1)

        t_user_rep = torch.cat(
            (
                torch.unsqueeze(self.t_rep, 2)[: self.n_users],
                torch.unsqueeze(self.t_rep, 2)[: self.n_users],
            ),
            dim=2,
        )
        t_user_rep = torch.cat((t_user_rep[:, :, 0], t_user_rep[:, :, 1]), dim=1)

        user_rep_n1 = torch.cat(
            (
                torch.unsqueeze(self.v_rep_n1, 2)[: self.n_users],
                torch.unsqueeze(self.t_rep_n1, 2)[: self.n_users],
            ),
            dim=2,
        )
        user_rep_n1 = self.weight_u.transpose(1, 2) * user_rep_n1
        user_rep_n1 = torch.cat((user_rep_n1[:, :, 0], user_rep_n1[:, :, 1]), dim=1)

        user_rep_n2 = torch.cat(
            (
                torch.unsqueeze(self.v_rep_n2, 2)[: self.n_users],
                torch.unsqueeze(self.t_rep_n2, 2)[: self.n_users],
            ),
            dim=2,
        )
        user_rep_n2 = self.weight_u.transpose(1, 2) * user_rep_n2
        user_rep_n2 = torch.cat((user_rep_n2[:, :, 0], user_rep_n2[:, :, 1]), dim=1)

        item_rep = representation[self.n_users :] + self.buildItemGraph(
            representation[self.n_users :]
        )
        guide_item_rep = guide_representation[self.n_users :] + self.buildItemGraph(
            guide_representation[self.n_users :]
        )
        v_item_rep = v_representation[self.n_users :] + self.buildItemGraph(
            v_representation[self.n_users :]
        )
        t_item_rep = t_representation[self.n_users :] + self.buildItemGraph(
            t_representation[self.n_users :]
        )
        item_rep_n1 = representation_n1[self.n_users :] + self.buildItemGraph(
            representation_n1[self.n_users :]
        )
        item_rep_n2 = representation_n2[self.n_users :] + self.buildItemGraph(
            representation_n2[self.n_users :]
        )

        self.user_rep, self.item_rep = user_rep, item_rep
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        self.result_embed_guide = torch.cat((guide_user_rep, guide_item_rep), dim=0)
        self.result_embed_v = torch.cat((v_user_rep, v_item_rep), dim=0)
        self.result_embed_t = torch.cat((t_user_rep, t_item_rep), dim=0)
        self.result_embed_n1 = torch.cat((user_rep_n1, item_rep_n1), dim=0)
        self.result_embed_n2 = torch.cat((user_rep_n2, item_rep_n2), dim=0)

        pos_scores = torch.sum(
            self.result_embed[user_nodes] * self.result_embed[pos_item_nodes], dim=1
        )
        neg_scores = torch.sum(
            self.result_embed[user_nodes] * self.result_embed[neg_item_nodes], dim=1
        )
        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        pos_scores, neg_scores = self.forward(interaction)

        loss_value = -torch.log2(torch.sigmoid(pos_scores - neg_scores)).mean()

        reg_embedding_loss_v = (
            (self.v_preference[user] ** 2).mean()
            if self.v_preference is not None
            else 0.0
        )
        reg_embedding_loss_t = (
            (self.t_preference[user] ** 2).mean()
            if self.t_preference is not None
            else 0.0
        )
        reg_loss = (
            self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
            + self.reg_weight * (self.weight_u**2).mean()
        )

        u_temp = self.user_rep[user].detach()
        i_temp = self.item_rep[pos_item].detach()
        u_temp2 = self.mlp(u_temp)
        i_temp2 = self.mlp(i_temp)
        u_temp_drop = F.dropout(u_temp, self.dropout)
        i_temp_drop = F.dropout(i_temp, self.dropout)

        mask_loss_u = 1 - F.cosine_similarity(u_temp, u_temp2).mean()
        mask_loss_i = 1 - F.cosine_similarity(i_temp, i_temp2).mean()
        mask_f_loss = self.mask_weight_f * (mask_loss_i + mask_loss_u)

        r_var, r_mean, g_var, g_mean, v_var, v_mean, t_var, t_mean = (
            self.fit_Gaussian_dis()
        )
        align_loss = (
            (torch.abs(g_var - r_var) + torch.abs(g_mean - r_mean)).mean()
            + (torch.abs(g_var - v_var) + torch.abs(g_mean - v_mean)).mean()
            + (torch.abs(g_var - t_var) + torch.abs(g_mean - t_mean)).mean()
            + (torch.abs(r_var - v_var) + torch.abs(r_mean - v_mean)).mean()
            + (torch.abs(r_var - t_var) + torch.abs(r_mean - t_mean)).mean()
            + (torch.abs(v_var - t_var) + torch.abs(v_mean - t_mean)).mean()
        )

        pos_item_nodes = pos_item + self.n_users
        mask_g_loss = self.InfoNCE(
            self.result_embed_n1[user],
            self.result_embed_n2[user],
            self.temp,
        ) + self.InfoNCE(
            self.result_embed_n1[pos_item_nodes],
            self.result_embed_n2[pos_item_nodes],
            self.temp,
        )

        return (
            loss_value
            + reg_loss
            + align_loss * self.align_weight
            + mask_f_loss
            + mask_g_loss * self.mask_weight_g
        )

    def fit_Gaussian_dis(self):
        return (
            torch.var(self.result_embed),
            torch.mean(self.result_embed),
            torch.var(self.result_embed_guide),
            torch.mean(self.result_embed_guide),
            torch.var(self.result_embed_v),
            torch.mean(self.result_embed_v),
            torch.var(self.result_embed_t),
            torch.mean(self.result_embed_t),
        )


class MENTOR(Recommender):
    def __init__(
        self,
        name="MENTOR",
        n_epochs=50,
        batch_size=512,
        embedding_size=64,
        # feat_embed_dim=64,
        n_mm_layers=1,
        knn_k=10,
        mm_image_weight=0.1,
        lr=0.0001,
        reg_weight=0.001,
        dropout=0.5,
        mask_weight_f=1.5,
        mask_weight_g=0.0001,
        align_weight=0.1,
        temp=0.2,
        device="cpu",
        verbose=True,
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)
        self.config = {
            "embedding_size": embedding_size,
            # "feat_embed_dim": feat_embed_dim,
            "n_mm_layers": n_mm_layers,
            "knn_k": knn_k,
            "mm_image_weight": mm_image_weight,
            "dropout": dropout,
            "reg_weight": reg_weight,
            "align_weight": align_weight,
            "mask_weight_g": mask_weight_g,
            "mask_weight_f": mask_weight_f,
            "temp": temp,
        }
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        v_numpy = train_set.item_image.features[: train_set.num_items]
        t_numpy = train_set.item_text.features[: train_set.num_items]
        v_feat = torch.tensor(v_numpy, dtype=torch.float32).to(self.device)
        t_feat = torch.tensor(t_numpy, dtype=torch.float32).to(self.device)

        R = train_set.matrix.tocoo()
        rows = R.row
        cols = R.col + train_set.num_users
        raw_edges = torch.tensor(np.vstack((rows, cols)), device=self.device, dtype=torch.long)

        self.raw_mentor = RawMENTORNet(
            self.config,
            train_set.num_users,
            train_set.num_items,
            v_feat,
            t_feat,
            raw_edges,
            self.device,
        ).to(self.device)
        optimizer = optim.Adam(self.raw_mentor.parameters(), lr=self.lr)

        from tqdm.auto import tqdm

        for epoch in range(1, self.n_epochs + 1):
            self.raw_mentor.train()
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
                interaction_tensors = (
                    torch.tensor(batch_users, dtype=torch.long).to(self.device),
                    torch.tensor(batch_pos_items, dtype=torch.long).to(self.device),
                    torch.tensor(batch_neg_items, dtype=torch.long).to(self.device),
                )
                loss = self.raw_mentor.calculate_loss(interaction_tensors)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                count += len(batch_users)
                progress_bar.set_postfix(loss=sum_loss / count * 100)
                progress_bar.update(1)
            progress_bar.close()

        self.raw_mentor.eval()
        with torch.no_grad():
            dummy = (
                torch.zeros(1, dtype=torch.long).to(self.device),
                torch.zeros(1, dtype=torch.long).to(self.device),
                torch.zeros(1, dtype=torch.long).to(self.device),
            )
            self.raw_mentor.forward(dummy)
            final_embed = self.raw_mentor.result_embed.cpu().numpy()

            trained_u_emb = final_embed[: train_set.num_users]
            trained_i_emb = final_embed[train_set.num_users :]

            self.cornac_user_embeddings = np.zeros(
                (self.total_users, trained_u_emb.shape[1])  # type: ignore
            )
            self.cornac_user_embeddings[: train_set.num_users] = trained_u_emb

            self.cornac_item_embeddings = np.zeros(
                (self.total_items, trained_i_emb.shape[1])  # type: ignore
            )
            self.cornac_item_embeddings[: train_set.num_items] = trained_i_emb

        return self

    def score(self, user_idx, item_idx=None):
        if item_idx is None:
            return np.dot(
                self.cornac_item_embeddings, self.cornac_user_embeddings[user_idx]
            )
        return np.dot(
            self.cornac_user_embeddings[user_idx], self.cornac_item_embeddings[item_idx]
        )
