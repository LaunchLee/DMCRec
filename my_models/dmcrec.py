"""Keep the original public class import pointed at the corrected model.

The selected paper settings are recorded in
experiments/dmcrec/configs/paper_configurations.json. Use the reproduction
entry point to replay them instead of relying on constructor defaults.
"""
from experiments.dmcrec.my_models.myrec import MyRec

DMCRec = MyRec
