import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from cornac.utils import get_rng
from cornac.utils.init_utils import xavier_uniform, zeros
from cornac.models import Recommender
from cornac.data import Dataset


class DMCRec(Recommender):
    def __init__(
        self,
        name="DMCRec",
        k=10,
        k2=10,
        n_epochs=50,
        batch_size=128,
        learning_rate=0.001,
        lambda_w=0.001,
        lambda_e=0.001,
        lambda_cl=0.01,
        cl_tau=0.1,
        cl_multi=True,
        use_knn_graph=True,
        knn_k=10,
        epoch_refresh_knn=40,
        epoch_refresh_knn_auto=False,
        epoch_refresh_knn_auto_delay=False,
        epoch_refresh_knn_auto_delay_flag=5,
        knn_sim_threshold=0.92,
        cl_multi_next=True,
        use_debias=True,
        macr_alpha=0.5,
        macr_beta=0.5,
        use_gcn=True,
        n_layers=2,
        lighter_perturb=False,
        edge_dropout_rate=0.1,
        node_dropout_rate=0.1,
        feature_dropout_rate=0.2,
        mask_weight=0.1,
        use_text=True,
        text_dim=768,
        use_scheduler=False,
        use_gpu=False,
        verbose=True,
        seed=None,
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)

        # BPR & Embeddings
        self.k = k
        self.k2 = k2
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_w = lambda_w
        self.lambda_e = lambda_e

        # Contrastive Learning
        self.lambda_cl = lambda_cl
        self.cl_tau = cl_tau
        self.cl_multi = cl_multi
        self.cl_multi_next = cl_multi_next

        # KNN Graph
        self.use_knn_graph = use_knn_graph
        self.knn_k = knn_k
        self.epoch_refresh_knn = epoch_refresh_knn
        self.epoch_refresh_knn_auto = epoch_refresh_knn_auto
        self.epoch_refresh_knn_auto_delay = epoch_refresh_knn_auto_delay
        self.epoch_refresh_knn_auto_delay_flag = epoch_refresh_knn_auto_delay_flag
        self.knn_sim_threshold = knn_sim_threshold

        # MACR (Causal Debias)
        self.use_debias = use_debias
        self.macr_alpha = macr_alpha
        self.macr_beta = macr_beta

        # Graph Convolutions & Dropouts
        self.use_gcn = use_gcn
        self.n_layers = n_layers
        self.lighter_perturb = lighter_perturb
        self.edge_dropout_rate = edge_dropout_rate
        self.node_dropout_rate = node_dropout_rate
        self.feature_dropout_rate = feature_dropout_rate
        self.mask_weight = mask_weight

        # Multimodal
        self.use_text = use_text
        self.text_dim = text_dim

        # General
        self.use_scheduler = use_scheduler
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.seed = seed

    def _init(self, n_users, n_items, visual_dim):
        rng = get_rng(self.seed)
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        # Collaborative Filtering Embeddings
        self.beta_item = zeros(n_items)
        self.gamma_user = xavier_uniform((n_users, self.k), rng)
        self.gamma_item = xavier_uniform((n_items, self.k), rng)
        self.theta_user = xavier_uniform((n_users, self.k2), rng)

        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        # Modality Projections
        self.visual_proj = nn.Linear(visual_dim, self.k2)
        if self.use_text:
            self.text_proj = nn.Linear(self.text_dim, self.k2)
            self.visual_norm = nn.LayerNorm(visual_dim)
            self.text_norm = nn.LayerNorm(self.text_dim)

        if self.cl_multi:
            self.id_proj = nn.Linear(self.k, self.k2)

        # Causal Debias: Popularity Bias Branch
        self.item_pop_bias = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.item_pop_bias.weight)

        # Feature Reconstruction (Masking)
        self.feature_mlp = nn.Sequential(
            nn.Linear(self.k2, self.k2), nn.ReLU(), nn.Linear(self.k2, self.k2)
        )
        self.feature_dropout = nn.Dropout(self.feature_dropout_rate)

        # Modality Adaptive Attention Weights
        self.alpha_v = nn.Parameter(torch.tensor(0.5))
        self.alpha_t = nn.Parameter(torch.tensor(0.5))

    def _build_knns_graph(self, features, chunk_size=4096):
        with torch.no_grad():
            N = self.total_items
            norm_feats = F.normalize(features, p=2, dim=1)

            topk_sim_list = []
            indices_list = []

            for i in range(0, N, chunk_size):  # type: ignore
                chunk = norm_feats[i : i + chunk_size]
                sim_chunk = torch.matmul(chunk, norm_feats.t())
                topk_sim, indices = torch.topk(sim_chunk, k=self.knn_k, dim=-1)

                topk_sim_list.append(topk_sim)
                indices_list.append(indices)

                del chunk, sim_chunk

            topk_sim_all = torch.cat(topk_sim_list, dim=0)
            indices_all = torch.cat(indices_list, dim=0)

            row = torch.arange(N, device=self.device).view(-1, 1).expand(-1, self.knn_k).flatten()  # type: ignore
            col = indices_all.flatten()

            weights = topk_sim_all.flatten() + 1e-7
            ii_indices = torch.stack([row, col], dim=0)

            row_sum = torch.zeros(N, device=self.device, dtype=torch.float32)  # type: ignore
            row_sum.scatter_add_(0, row, weights)
            row_sum = row_sum + 1e-7

            r_inv_sqrt = torch.pow(row_sum, -0.5)
            values = r_inv_sqrt[ii_indices[0]] * r_inv_sqrt[ii_indices[1]]

            with torch.sparse.check_sparse_tensor_invariants():
                ret = torch.sparse_coo_tensor(ii_indices, values, (N, N), device=self.device)  # type: ignore

            del norm_feats, topk_sim_list, indices_list, topk_sim_all, indices_all, row, col, weights, row_sum, r_inv_sqrt
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return ret

    def _cal_infonce_loss(self, embed_1, embed_2):
        embed_1 = F.normalize(embed_1, p=2, dim=1)
        embed_2 = F.normalize(embed_2, p=2, dim=1)
        pos = torch.sum(embed_1 * embed_2, dim=-1)
        tot = torch.matmul(embed_1, embed_2.t())
        logits = tot - pos[:, None]
        return torch.mean(torch.logsumexp(logits / self.cl_tau, dim=1))

    def sparse_dropout(self, mat, dropout_ratio):
        if dropout_ratio <= 0.0:
            return mat
        indices = mat._indices()
        values = mat._values()
        mask = torch.rand(values.size()[0], device=mat.device) > dropout_ratio
        new_indices = indices[:, mask]
        new_values = values[mask] / (1.0 - dropout_ratio)
        with torch.sparse.check_sparse_tensor_invariants():
            ret = torch.sparse_coo_tensor(new_indices, new_values, mat.size(), device=mat.device)
        return ret

    def get_embeddings(self, norm_adj, Gu, Gi, is_train=True, perturbed=False):
        ego_embeddings = torch.cat([Gu, Gi], dim=0)

        out = ego_embeddings.clone()
        curr = ego_embeddings

        if is_train and self.edge_dropout_rate > 0.0:
            adj = self.sparse_dropout(norm_adj, self.edge_dropout_rate)
        else:
            adj = norm_adj

        if is_train and perturbed and self.node_dropout_rate > 0.0:
            random_noise = torch.rand_like(curr)
            curr = curr + torch.sign(curr) * F.normalize(random_noise, dim=-1) * 0.1

        for _ in range(self.n_layers):
            curr = torch.sparse.mm(adj, curr)
            if is_train and perturbed and self.node_dropout_rate > 0.0:
                random_noise = torch.rand_like(curr)
                curr = curr + torch.sign(curr) * F.normalize(random_noise, dim=-1) * 0.1
            out = out + curr

        out = out / (self.n_layers + 1.0)
        g_u_gcn, g_i_gcn = torch.split(out, [self.total_users, self.total_items], dim=0)  # type: ignore
        return g_u_gcn, g_i_gcn

    def build_ui_graph(self, train_set: Dataset):
        n_users = train_set.num_users
        n_items = train_set.num_items
        R = train_set.matrix

        top_block = sp.hstack([sp.csr_matrix((n_users, n_users)), R])
        bottom_block = sp.hstack([R.T, sp.csr_matrix((n_items, n_items))])
        A = sp.vstack([top_block, bottom_block], format="coo")

        # GNN Laplace Normalize: D^(-1/2) * A * D^(-1/2)
        deg = np.array(A.sum(axis=1)).flatten()
        deg[deg == 0] = 1e-12
        d_inv = np.power(deg, -0.5)
        D_mat = sp.diags(d_inv)
        L_matrix = D_mat.dot(A).dot(D_mat).tocoo()

        indices = torch.LongTensor(np.vstack((L_matrix.row, L_matrix.col)))
        values = torch.FloatTensor(L_matrix.data)
        shape = torch.Size(L_matrix.shape)
        with torch.sparse.check_sparse_tensor_invariants():
            ret = torch.sparse_coo_tensor(indices, values, shape)
        return ret

    def fit(self, train_set, val_set=None):
        if self.verbose:
            print("[INFO] Model Name:", self.name)

        Recommender.fit(self, train_set, val_set)
        train_features_v = train_set.item_image.features[: self.total_items].astype(np.float32)
        visual_dim = train_features_v.shape[1]

        if self.use_text:
            self.text_dim = train_set.item_text.features.shape[1]

        self._init(n_users=self.total_users, n_items=self.total_items, visual_dim=visual_dim)
        self._fit_torch(train_set, train_features_v)
        return self

    def _fit_torch(self, train_set: Dataset, train_features_v: np.ndarray):
        device = self.device
        dtype = torch.float

        if self.use_gcn:
            norm_adj = self.build_ui_graph(train_set).to(device)

        V = torch.tensor(train_features_v, device=device, dtype=dtype)
        T = torch.tensor(train_set.item_text.features[: self.total_items], device=device, dtype=dtype) if self.use_text else None  # type: ignore

        Bi = torch.tensor(self.beta_item, device=device, dtype=dtype, requires_grad=True)
        Gu = torch.tensor(self.gamma_user, device=device, dtype=dtype, requires_grad=True)
        Gi = torch.tensor(self.gamma_item, device=device, dtype=dtype, requires_grad=True)
        Tu = torch.tensor(self.theta_user, device=device, dtype=dtype, requires_grad=True)

        modules: list = [self.visual_proj, self.feature_mlp]
        if self.use_text:
            modules.extend([self.text_proj, self.visual_norm, self.text_norm])
        if self.use_debias:
            modules.append(self.item_pop_bias)
        if self.cl_multi:
            modules.append(self.id_proj)

        params = [Bi, Gu, Gi, Tu]
        if self.use_text:
            params.extend([self.alpha_v, self.alpha_t])

        for m in modules:
            m.to(device)
            m.train()
            params.extend(list(m.parameters()))

        if self.use_knn_graph:
            if self.verbose:
                print("[INFO] Pre-building Item KNN graphs for Multimodal Broadcasting...")
            with torch.no_grad():
                v_init = self.visual_proj(self.visual_norm(V) if self.use_text else V)
                v_knn_adj = self._build_knns_graph(v_init)
                v_feat_last = v_init.detach().clone()
                if self.use_text and T is not None:
                    t_init = self.text_proj(self.text_norm(T))
                    t_knn_adj = self._build_knns_graph(t_init)
                    t_feat_last = t_init.detach().clone()

        knn_updated_times = {"V": 0, "T": 0}
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-5)

        refresh_knn = not self.epoch_refresh_knn_auto_delay
        if self.use_knn_graph and not refresh_knn and self.verbose:
            print("[INFO] KNN graph is freezing for now.")

        last_avg_loss = 0.0
        max_improv = 1e-8
        low_improv_flag = 0

        from tqdm.auto import tqdm

        for epoch in range(1, self.n_epochs + 1):
            sum_loss = 0.0
            count = 0
            progress_bar = tqdm(total=train_set.num_batches(self.batch_size), desc=f"Epoch {epoch}/{self.n_epochs}", disable=not self.verbose)

            # ---------------- KNN Auto-Refresh Logic ----------------
            if self.use_knn_graph and refresh_knn and epoch > 1:
                if self.epoch_refresh_knn_auto:
                    with torch.no_grad():
                        current_v = self.visual_proj(self.visual_norm(V) if self.use_text else V)
                        sim = F.cosine_similarity(current_v, v_feat_last, dim=-1).mean().item()
                        if sim < self.knn_sim_threshold:
                            knn_updated_times["V"] += 1
                            if self.verbose:
                                print(f"\n[INFO] V feature drift detected (Sim={sim:.4f}). Re-building KNN...({knn_updated_times['V']})")
                            v_knn_adj = self._build_knns_graph(current_v)
                            v_feat_last = current_v.detach().clone()
                        if self.use_text and T is not None:
                            current_t = self.text_proj(self.text_norm(T))
                            sim = F.cosine_similarity(current_t, t_feat_last, dim=-1).mean().item()
                            if sim < self.knn_sim_threshold:
                                knn_updated_times["T"] += 1
                                if self.verbose:
                                    print(f"[INFO] T feature drift detected (Sim={sim:.4f}). Re-building KNN...({knn_updated_times['T']})")
                                t_knn_adj = self._build_knns_graph(current_t)
                                t_feat_last = current_t.detach().clone()
                elif (self.epoch_refresh_knn and self.epoch_refresh_knn > 0 and (self.epoch_refresh_knn == 1 or epoch % self.epoch_refresh_knn == 1)):
                    if self.verbose:
                        print(f"\n[INFO] Re-building graphs at epoch {epoch} using updated projections...")
                    with torch.no_grad():
                        current_v = self.visual_proj(self.visual_norm(V) if self.use_text else V)
                        v_knn_adj = self._build_knns_graph(current_v)
                        knn_updated_times["V"] += 1
                        if self.use_text and T is not None:
                            current_t = self.text_proj(self.text_norm(T))
                            t_knn_adj = self._build_knns_graph(current_t)
                            knn_updated_times["T"] += 1
            # --------------------------------------------------------

            for batch_u, batch_i, batch_j in train_set.uij_iter(self.batch_size, shuffle=True):
                optimizer.zero_grad()

                bu_t = torch.tensor(batch_u, dtype=torch.long, device=device)
                bi_t = torch.tensor(batch_i, dtype=torch.long, device=device)
                bj_t = torch.tensor(batch_j, dtype=torch.long, device=device)

                if self.use_knn_graph:
                    gi_v = torch.sparse.mm(v_knn_adj, Gi)
                    if self.use_text and T is not None:
                        gi_t = torch.sparse.mm(t_knn_adj, Gi)
                        Gi_fused = Gi + 0.5 * gi_v + 0.5 * gi_t
                    else:
                        Gi_fused = Gi + gi_v
                else:
                    Gi_fused = Gi

                # 1. User-Item Graph Embeddings
                if self.use_gcn:
                    g_u_gcn, g_i_gcn = self.get_embeddings(norm_adj, Gu, Gi_fused, is_train=True, perturbed=False)  # type: ignore
                else:
                    g_u_gcn, g_i_gcn = Gu, Gi_fused

                g_u, g_i, g_j = g_u_gcn[bu_t], g_i_gcn[bi_t], g_i_gcn[bj_t]
                gu_0, gi_0, gj_0 = Gu[bu_t], Gi[bi_t], Gi[bj_t]
                b_i, b_j = Bi[bi_t], Bi[bj_t]

                # 2. Multimodal KNN Graph Smoothing
                base_v = self.visual_proj(self.visual_norm(V) if self.use_text else V)
                v_feat_all = base_v + torch.sparse.mm(v_knn_adj, base_v) if self.use_knn_graph else base_v

                if self.use_text and T is not None:
                    weight_v = torch.exp(self.alpha_v) / (torch.exp(self.alpha_v) + torch.exp(self.alpha_t))
                    weight_t = torch.exp(self.alpha_t) / (torch.exp(self.alpha_v) + torch.exp(self.alpha_t))
                    base_t = self.text_proj(self.text_norm(T))
                    t_feat_all = base_t + torch.sparse.mm(t_knn_adj, base_t) if self.use_knn_graph else base_t
                    item_feat_all = F.normalize(weight_v * v_feat_all + weight_t * t_feat_all, p=2, dim=1)
                else:
                    item_feat_all = F.normalize(v_feat_all, p=2, dim=1)

                # 3. Propagate Multimodal Features on User-Item Graph
                if self.use_gcn:
                    f_u_gcn, f_i_gcn = self.get_embeddings(norm_adj, Tu, item_feat_all, is_train=True, perturbed=False)  # type: ignore
                else:
                    f_u_gcn, f_i_gcn = Tu, item_feat_all

                t_u = f_u_gcn[bu_t]
                item_feat_i = f_i_gcn[bi_t]
                item_feat_j = f_i_gcn[bj_t]

                # 4. Score Calculation (Main Branch)
                i_m = b_i + (g_u * g_i).sum(dim=1) + (t_u * item_feat_i).sum(dim=1)
                j_m = b_j + (g_u * g_j).sum(dim=1) + (t_u * item_feat_j).sum(dim=1)

                # 5. Causal Debias (Popularity Bias Branch)
                if self.use_debias:
                    i_n = self.item_pop_bias(bi_t).squeeze(-1)
                    j_n = self.item_pop_bias(bj_t).squeeze(-1)

                # Fused scores (Multi-task causal reasoning)
                if self.use_debias:
                    i_score = i_m * torch.sigmoid(i_n)
                    j_score = j_m * torch.sigmoid(j_n)
                else:
                    i_score = i_m
                    j_score = j_m

                # Total BPR Loss
                loss_bpr = -F.logsigmoid(i_score - j_score).sum()

                if self.use_debias:
                    loss_bpr = (
                        -F.logsigmoid(i_m - j_m).sum() * self.macr_alpha
                        + -F.logsigmoid(i_n - j_n).sum() * self.macr_beta  # type: ignore
                    )

                # 6. Contrastive Learning Loss (Conditional Execution)
                loss_cl = 0.0
                if self.lambda_cl > 0:
                    if self.use_gcn:
                        if not self.lighter_perturb:
                            # Perturbation for Contrastive Learning
                            _, g_i_gcn_n1 = self.get_embeddings(norm_adj, Gu, Gi_fused, is_train=True, perturbed=True)  # type: ignore
                            _, g_i_gcn_n2 = self.get_embeddings(norm_adj, Gu, Gi_fused, is_train=True, perturbed=True)  # type: ignore
                            g_u_gcn_n1, _ = self.get_embeddings(norm_adj, Gu, Gi_fused, is_train=True, perturbed=True)  # type: ignore
                            g_u_gcn_n2, _ = self.get_embeddings(norm_adj, Gu, Gi_fused, is_train=True, perturbed=True)  # type: ignore

                            loss_cl_g_u = self._cal_infonce_loss(g_u_gcn_n1[bu_t], g_u_gcn_n2[bu_t])
                            loss_cl_g_i = self._cal_infonce_loss(g_i_gcn_n1[bi_t], g_i_gcn_n2[bi_t])
                            loss_cl += (loss_cl_g_u + loss_cl_g_i) * len(bu_t)
                        else:
                            # Add Gaussian noise to embeddings
                            noise_u1 = torch.randn_like(g_u) * 0.1
                            noise_u2 = torch.randn_like(g_u) * 0.1
                            noise_i1 = torch.randn_like(g_i) * 0.1
                            noise_i2 = torch.randn_like(g_i) * 0.1

                            loss_cl_g_u = self._cal_infonce_loss(g_u + noise_u1, g_u + noise_u2)
                            loss_cl_g_i = self._cal_infonce_loss(g_i + noise_i1, g_i + noise_i2)
                            loss_cl += (loss_cl_g_u + loss_cl_g_i) * len(bu_t)

                    v_feat_i = v_feat_all[bi_t]
                    if self.use_text and T is not None:
                        t_feat_i = t_feat_all[bi_t]
                    item_feat_i_raw = item_feat_all[bi_t]

                    if self.use_text and T is not None:
                        if self.cl_multi_next:
                            id_feat_i = self.id_proj(g_i)
                            loss_cl_v = self._cal_infonce_loss(id_feat_i, v_feat_i)
                            loss_cl_t = self._cal_infonce_loss(id_feat_i, t_feat_i)
                            loss_cl_f = self._cal_infonce_loss(id_feat_i, item_feat_i_raw)
                            loss_cl += (loss_cl_v + loss_cl_t + loss_cl_f) * len(bu_t)
                        else:
                            norm_t = nn.functional.normalize(t_feat_i, dim=1)  # type: ignore
                            norm_v = nn.functional.normalize(v_feat_i, dim=1)
                            labels = torch.arange(norm_v.size(0), device=device)

                            logits_vt = norm_t @ norm_v.t() / self.cl_tau
                            loss_l1 = nn.functional.cross_entropy(logits_vt, labels)
                            if self.cl_multi:
                                id_feat_i = self.id_proj(g_i)
                                norm_id = nn.functional.normalize(id_feat_i, dim=1)
                                norm_f = nn.functional.normalize(item_feat_i_raw, dim=1)

                                logits_vid = norm_v @ norm_id.t() / self.cl_tau
                                logits_tid = norm_t @ norm_id.t() / self.cl_tau
                                loss_l2 = 0.5 * (nn.functional.cross_entropy(logits_vid, labels) + nn.functional.cross_entropy(logits_tid, labels))

                                logits_fid = norm_f @ norm_id.t() / self.cl_tau
                                loss_f_id = nn.functional.cross_entropy(logits_fid, labels)
                                logits_f_v = norm_v @ norm_f.detach().t() / self.cl_tau
                                loss_f_v = nn.functional.cross_entropy(logits_f_v, labels)
                                logits_f_t = norm_t @ norm_f.detach().t() / self.cl_tau
                                loss_f_t = nn.functional.cross_entropy(logits_f_t, labels)

                                loss_l3 = loss_f_id + 0.1 * (loss_f_v + loss_f_t)
                                loss_cl += loss_l1 + loss_l2 + loss_l3
                            else:
                                loss_cl += loss_l1

                # 7. Masking Loss (Conditional Execution)
                loss_mask = 0.0
                if self.mask_weight > 0 and self.use_text and T is not None:
                    # Using item_feat_i (which contains GCN propagated multimodalities)
                    item_feat_i_drop = self.feature_dropout(item_feat_i)
                    item_feat_i_recon = self.feature_mlp(item_feat_i_drop)
                    loss_mask += (1 - F.cosine_similarity(item_feat_i_drop, item_feat_i_recon).mean()) * len(bu_t)

                # 8. Reg and Optimization
                reg_w = (gu_0.pow(2).sum() + gi_0.pow(2).sum() + gj_0.pow(2).sum()) / 2.0
                reg_n = (sum(p.pow(2).sum() for m in modules for p in m.parameters()) / 2.0)

                loss = (
                    loss_bpr
                    + self.lambda_w * reg_w
                    + self.lambda_e * reg_n
                    + self.lambda_cl * loss_cl
                    + self.mask_weight * loss_mask
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                count += len(bu_t)
                epoch_avg_loss = sum_loss / count
                progress_bar.set_postfix(loss=epoch_avg_loss)
                progress_bar.update(1)

            progress_bar.close()
            if self.use_scheduler:
                scheduler.step()

            # Loss Plateau Monitoring -> KNN auto refresh
            if self.epoch_refresh_knn_auto_delay and not refresh_knn:
                if epoch > 1:
                    improv = last_avg_loss - epoch_avg_loss
                    max_improv = max((improv + 1e-8), max_improv)
                    if (improv / max_improv) < 0.03:
                        low_improv_flag += 1
                        if low_improv_flag > self.epoch_refresh_knn_auto_delay_flag:
                            refresh_knn = True
                            if self.use_knn_graph and self.verbose:
                                print(f"\n[INFO] Loss plateaued at epoch {epoch}. KNN is now allowed to refresh.")
                    else:
                        low_improv_flag = max(0, low_improv_flag - 1)
                last_avg_loss = epoch_avg_loss

        if self.use_knn_graph and self.verbose:
            print(f"[INFO] KNN graph updated times: {knn_updated_times}")

        # ---------------- Baking Phase ----------------
        self.beta_item = Bi.detach().cpu().numpy().copy()
        self.gamma_user = Gu.detach().cpu().numpy().copy()
        self.gamma_item = Gi.detach().cpu().numpy().copy()
        self.theta_user = Tu.detach().cpu().numpy().copy()

        for m in modules:
            m.eval()

        with torch.no_grad():
            # Testing Stage Gi smoothing
            if self.use_knn_graph:
                gi_v_eval = torch.sparse.mm(v_knn_adj, Gi.detach())
                if self.use_text and T is not None:
                    gi_t_eval = torch.sparse.mm(t_knn_adj, Gi.detach())
                    Gi_fused_eval = Gi.detach() + 0.5 * gi_v_eval + 0.5 * gi_t_eval
                else:
                    Gi_fused_eval = Gi.detach() + gi_v_eval
            else:
                Gi_fused_eval = Gi.detach()

            if self.use_gcn:
                g_u_gcn_final, g_i_gcn_final = self.get_embeddings(norm_adj, Gu.detach(), Gi_fused_eval, is_train=False)  # type: ignore
                self.g_u_gcn = g_u_gcn_final.cpu().numpy().copy()
                self.g_i_gcn = g_i_gcn_final.cpu().numpy().copy()

            if self.use_knn_graph:
                base_v = self.visual_proj(self.visual_norm(V) if self.use_text else V)
                v_all = base_v + torch.sparse.mm(v_knn_adj, base_v)
                if self.use_text and T is not None:
                    weight_v = torch.exp(self.alpha_v) / (torch.exp(self.alpha_v) + torch.exp(self.alpha_t))
                    weight_t = torch.exp(self.alpha_t) / (torch.exp(self.alpha_v) + torch.exp(self.alpha_t))
                    base_t = self.text_proj(self.text_norm(T))
                    t_all = base_t + torch.sparse.mm(t_knn_adj, base_t)
                    feat_all = F.normalize(weight_v * v_all + weight_t * t_all, p=2, dim=1)
                else:
                    feat_all = F.normalize(v_all, p=2, dim=1)
            else:
                base_v = self.visual_proj(self.visual_norm(V) if self.use_text else V)
                v_all = base_v
                if self.use_text and T is not None:
                    weight_v = torch.exp(self.alpha_v) / (torch.exp(self.alpha_v) + torch.exp(self.alpha_t))
                    weight_t = torch.exp(self.alpha_t) / (torch.exp(self.alpha_v) + torch.exp(self.alpha_t))
                    base_t = self.text_proj(self.text_norm(T))
                    t_all = base_t
                    feat_all = F.normalize(weight_v * v_all + weight_t * t_all, p=2, dim=1)
                else:
                    feat_all = F.normalize(v_all, p=2, dim=1)

            if self.use_gcn:
                f_u_final, f_i_final = self.get_embeddings(norm_adj, Tu.detach(), feat_all.detach(), is_train=False)  # type: ignore
                self.f_u_gcn = f_u_final.cpu().numpy().copy()
                self.item_feat_baked = f_i_final.cpu().numpy().copy()
            else:
                self.f_u_gcn = self.theta_user
                self.item_feat_baked = feat_all.detach().cpu().numpy()

        if self.verbose:
            print("Optimization and baking finished!")

    def score(self, user_idx, item_idx=None):
        """Counterfactual Inference: Predict without Popularity Bias Branch"""
        g_u = self.g_u_gcn[user_idx] if self.use_gcn else self.gamma_user[user_idx]
        g_i = self.g_i_gcn if self.use_gcn else self.gamma_item
        t_u = self.f_u_gcn[user_idx] if self.use_gcn else self.theta_user[user_idx]

        if item_idx is not None:
            item_score = self.beta_item[item_idx] + np.dot(g_u, g_i[item_idx])
            if hasattr(self, "item_feat_baked"):
                item_score += np.dot(t_u, self.item_feat_baked[item_idx])
            return item_score
        else:
            m_score = self.beta_item + g_u @ g_i.T
            if hasattr(self, "item_feat_baked"):
                m_score += t_u @ self.item_feat_baked.T
            return m_score
