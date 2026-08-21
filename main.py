print("[INFO] Importing...")


from cornac.metrics import Recall, MRR, NDCG, HitRatio
from cornac import Experiment

from my_models.dmcrec import DMCRec

from data import *
from misc import *


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
CACHE_DIR = create_dir_if_not_exist("cache")
SEED = 42
seed_everything(SEED)


print("[INFO] Preparing...")
ratio_split, mm_visuals_mean = load_cornac_clothing(seed=SEED)


NUM_FACTOR1 = 128
NUM_FACTOR2 = 64
NUM_FACTOR3 = 32
NUM_EPOCHS = 200
BATCH_SIZE = 512
VERBOSE = True

MODELS_SWITCH = {
    "DMCRec_Latest": True
}

experiment_models = []

# models ---------------------------------------------------------------------
if MODELS_SWITCH["DMCRec_Latest"]:
    myRec = DMCRec(
        name="MyRec04",  # The name is MyRec04
        k=NUM_FACTOR1,
        k2=NUM_FACTOR2,
        n_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,

        learning_rate=0.001,

        lambda_w=0.001,
        lambda_e=0.001,

        lambda_cl=0.01,
        cl_tau=0.2,
        cl_multi=True,

        use_knn_graph=True,
        knn_k=10,
        epoch_refresh_knn=0,
        epoch_refresh_knn_auto=True,
        epoch_refresh_knn_auto_delay=True,
        epoch_refresh_knn_auto_delay_flag=10,
        knn_sim_threshold=0.85,
        cl_multi_next=True,

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

        use_scheduler=False,
        use_gpu=True,
        verbose=VERBOSE,
        seed=SEED
    )
    experiment_models.append(myRec)

# ------------------------------------------------------------------------------
if experiment_models:
    Experiment(
        eval_method=ratio_split,
        models=experiment_models,
        metrics=[
            HitRatio(50),
            MRR(),
            NDCG(20),
            NDCG(50),
            Recall(20),
            Recall(50),
        ],
        save_dir=os.path.join(CACHE_DIR, 'Group101')
    ).run()
else:
    print("No models selected for experiment.")
