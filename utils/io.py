from rdkit import Chem
import pandas as pd
from pathlib import Path
# import yaml
from fvcore.common.config import CfgNode

def load_from_csv(csv_fn, smiles_column='smiles'):
    mols = []
    for _, row in pd.read_csv(csv_fn).iterrows():
        smiles = row[smiles_column]
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
    return mols

# add by 
def load_smiles_fn(smiles_fn):
    read_params={
        '.csv': {'header': 0, 'sep':','},
        '.smi': {'header': None, 'sep':'\t'},
        '.tsv': {'header': 0, 'sep':'\t'},
    }
    smiles_strs = []
    suffix = Path(smiles_fn).suffix.lower()
    with open(smiles_fn, 'r') as f:
        reader = pd.read_csv(f, **read_params[suffix])
        for _, row in reader.iterrows():
            smiles_strs.append(row[0])
    return smiles_strs

# def load_config(config_fn):
#     with open(config_fn, 'r') as stream:
#         # config = yaml.safe_load(stream)
#         config = yaml.safe_load(stream)
#     return config

def load_config(config_fn) -> CfgNode:
    _cfg = CfgNode.load_yaml_with_base(config_fn)
    cfg: CfgNode = CfgNode(_cfg)
    return cfg