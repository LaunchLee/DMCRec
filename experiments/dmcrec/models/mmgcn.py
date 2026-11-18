import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cornac.models import Recommender

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class MMGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MMGCNConv, self).__init__(
            aggr="add"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, edge_index, norm):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class SingleModalGCN(nn.Module):
    def __init__(
        self,
        num_user,
        num_item,
        dim_feat,
        dim_id,
        num_layer=3,
        dim_latent=None,
        device="cpu",
    ):
        super(SingleModalGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.num_layer = num_layer
        self.dim_latent = dim_latent
        self.device = torch.device(device) if isinstance(device, str) else device

        if self.dim_latent:
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            nn.init.xavier_normal_(self.MLP.weight)
            in_dim = self.dim_latent
        else:
            self.MLP = None
            in_dim = self.dim_feat

        self.preference = nn.Parameter(torch.empty(num_user, in_dim))
        nn.init.xavier_normal_(self.preference)

        self.conv_embed_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.g_layers = nn.ModuleList()

        for i in range(self.num_layer):
            if i == 0:
                conv = MMGCNConv(in_dim, in_dim)
                linear = nn.Linear(in_dim, dim_id)
                g_layer = nn.Linear(in_dim, dim_id)
            else:
                conv = MMGCNConv(dim_id, dim_id)
                linear = nn.Linear(dim_id, dim_id)
                g_layer = nn.Linear(dim_id, dim_id)

            nn.init.xavier_normal_(linear.weight)
            nn.init.xavier_normal_(g_layer.weight)

            self.conv_embed_layers.append(conv)
            self.linear_layers.append(linear)
            self.g_layers.append(g_layer)

    def forward(self, features, id_embedding, edge_index, norm):
        features = F.normalize(features, p=2, dim=1)
        temp_features = self.MLP(features) if self.MLP is not None else features

        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x, p=2, dim=1)

        for i in range(self.num_layer):
            h = F.leaky_relu(self.conv_embed_layers[i](x, edge_index, norm))
            x_hat = F.leaky_relu(self.linear_layers[i](x)) + id_embedding
            if i == 0:
                x = F.leaky_relu(self.g_layers[i](h) + x_hat)
            else:
                x = F.leaky_relu(self.g_layers[i](h + x_hat))

        return x


class RawMMGCN(nn.Module):
    def __init__(
        self,
        num_user,
        num_item,
        edge_index,
        embedding_size,
        reg_weight,
        v_feat=None,
        t_feat=None,
        device="cpu",
    ):
        super(RawMMGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.n_nodes = num_user + num_item
        self.reg_weight = reg_weight
        self.device = device

        self.v_feat = v_feat
        self.t_feat = t_feat
        self.edge_index = edge_index

        with torch.no_grad():
            row, col = self.edge_index
            deg = degree(row, self.n_nodes, dtype=torch.float32)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
            self.norm = (deg_inv_sqrt[row] * deg_inv_sqrt[col]).to(self.device)

        self.id_embedding = nn.Parameter(torch.empty(self.n_nodes, embedding_size))
        nn.init.xavier_normal_(self.id_embedding)

        self.num_modal = 0
        if self.v_feat is not None:
            self.v_gcn = SingleModalGCN(
                num_user,
                num_item,
                self.v_feat.size(1),
                embedding_size,
                dim_latent=256,
                device=self.device,
            )
            self.num_modal += 1

        if self.t_feat is not None:
            self.t_gcn = SingleModalGCN(
                num_user,
                num_item,
                self.t_feat.size(1),
                embedding_size,
                dim_latent=None,
                device=self.device,
            )
            self.num_modal += 1

    def forward(self):
        representation = None
        if self.v_feat is not None:
            representation = self.v_gcn(self.v_feat, self.id_embedding, self.edge_index, self.norm)
        if self.t_feat is not None:
            if representation is None:
                representation = self.t_gcn(
                    self.t_feat, self.id_embedding, self.edge_index, self.norm
                )
            else:
                representation = representation + self.t_gcn(
                    self.t_feat, self.id_embedding, self.edge_index, self.norm
                )

        if representation is not None and self.num_modal > 0:
            representation = representation / self.num_modal
        else:
            representation = self.id_embedding

        return representation

    def calculate_loss(self, users, pos_items, neg_items):
        out = self.forward()

        pos_items_nodes = pos_items + self.num_user
        neg_items_nodes = neg_items + self.num_user

        u_emb = out[users]
        pos_emb = out[pos_items_nodes]
        neg_emb = out[neg_items_nodes]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        reg_loss_id = (
            (self.id_embedding[users] ** 2).mean()
            + (self.id_embedding[pos_items_nodes] ** 2).mean()
            + (self.id_embedding[neg_items_nodes] ** 2).mean()
        )

        reg_loss_modal = 0.0
        if self.v_feat is not None:
            reg_loss_modal += (self.v_gcn.preference**2).mean()
        if self.t_feat is not None:
            reg_loss_modal += (self.t_gcn.preference**2).mean()

        return bpr_loss + self.reg_weight * reg_loss_id + (self.reg_weight * 0.1) * reg_loss_modal


class MMGCN(Recommender):
    def __init__(
        self,
        name="MMGCN",
        embedding_size=64,
        reg_weight=1e-4,
        learning_rate=0.001,
        n_epochs=20,
        batch_size=1024,
        verbose=True,
        device="cpu",
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)
        self.embedding_size = embedding_size
        self.reg_weight = reg_weight
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device

        self.model = None
        self.final_result = None

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        R = train_set.matrix.tocoo()

        row_forward = R.row
        col_forward = R.col + train_set.num_users

        row_backward = R.col + train_set.num_users
        col_backward = R.row

        all_rows = np.concatenate([row_forward, row_backward])
        all_cols = np.concatenate([col_forward, col_backward])

        edge_index = torch.tensor(np.vstack((all_rows, all_cols)), device=self.device, dtype=torch.long)

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

        self.model = RawMMGCN(
            num_user=train_set.num_users,
            num_item=train_set.num_items,
            edge_index=edge_index,
            embedding_size=self.embedding_size,
            reg_weight=self.reg_weight,
            v_feat=v_feat,
            t_feat=t_feat,
            device=self.device  # type: ignore
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        num_ratings = train_set.num_ratings

        from tqdm.auto import tqdm

        for epoch in range(self.n_epochs):
            self.model.train()
            sum_loss = 0.0
            count = 0

            progress_bar = tqdm(
                total=train_set.num_batches(self.batch_size),
                desc=f"Epoch {epoch + 1}/{self.n_epochs}",
                disable=not self.verbose,
            )
            users_list, pos_items_list = train_set.matrix.nonzero()
            indices = np.arange(num_ratings)
            np.random.shuffle(indices)

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
            self.final_result = self.model().cpu().numpy()

    def score(self, user_idx, item_idx=None):
        user_tensor = self.final_result[: self.num_users]  # type: ignore
        item_tensor = self.final_result[self.num_users :]  # type: ignore

        u_emb = user_tensor[user_idx]
        if item_idx is None:
            return np.dot(u_emb, item_tensor.T)
        else:
            return np.dot(u_emb, item_tensor[item_idx].T)
