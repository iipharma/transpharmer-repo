from rdkit import Chem
from rdkit.Chem import Descriptors
import rdkit.RDLogger as rkl
rdlogger = rkl.logger()
rdlogger.setLevel(rkl.CRITICAL)
import time
import random

from utils.io import load_from_csv
from utils.pharm.misc import get_pharma_fp, get_pharma_count_fp, mapper


def get_props(csv_fn):
    props = set()
    mols = load_from_csv(csv_fn)
    for mol in mols:
        prop = get_pharma_fp(mol)
        prop = ''.join([str(t) for t in prop])
        props.add(prop)
    return props


def get_props_from_smiles(smiles_list, num_props):
    props = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            if num_props == 80:  # 72-bit fp & feature counts (8)
                prop = get_pharma_count_fp(mol)
            else:
                prop = get_pharma_fp(mol, n_dim=num_props)
            if prop not in props:
                props.append(prop)
    return props


def get_props_from_mols(mol_list, num_props):
    props = []
    for mol in mol_list:
        if num_props == 80:  # 72-bit fp & feature counts (8)
            prop = get_pharma_count_fp(mol)
        else:
            prop = get_pharma_fp(mol, n_dim=num_props)
        props.append(prop)
    return props


def cansmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            return Chem.MolToSmiles(mol, isomericSmiles = False, kekuleSmiles = True, canonical=True)
        except:
            return
    return


# add by xyj
def enumerate_fixed_scaf_smiles(smiles_, time_= 10):
    mol = Chem.MolFromSmiles(smiles_)
    id_fix = None
    # get dummy points
    for a_id, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() == 0:
            id_fix = a_id
            break
    
    id_not_fix = [t for t in range(mol.GetNumAtoms()) if t != id_fix]
    smiles_rand_all = set()
    s_t = time.time()
    # do random
    while True:
        random.shuffle(id_not_fix)
        id_all = id_not_fix + [id_fix]
        mol_rand = Chem.RenumberAtoms(mol, id_all)
        smiles_rand = Chem.MolToSmiles(mol_rand, canonical=False, isomericSmiles=True)
        if smiles_rand.endswith('*'):
            smiles_rand = smiles_rand[:-1]
        elif smiles_rand.endswith('[*]'):
            smiles_rand = smiles_rand[:-3]
        elif smiles_rand.endswith('(*)'):
            smiles_rand = smiles_rand[:-3]
        else:
            continue
        smiles_rand_all.add(smiles_rand)
        if (time.time()-s_t > time_) and (len(smiles_rand_all) > 1):
            break
    return smiles_rand_all


def cleanise_smiles(smiles, sanitize_mol=True):
    """a function to cleanise smiles into a starndard kekulized smiles string"""
    if len(smiles) == 0:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if sanitize_mol:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
    
    new_smiles = Chem.MolToSmiles(mol, isomericSmiles = False, kekuleSmiles = True, canonical=True)
    
    if len(new_smiles) == 0:
        return None

    return new_smiles


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


def double_check_valid(romol):
    flag = str(Chem.SanitizeMol(romol, catchErrors=True))
    if flag == "SANITIZE_NONE":
        smi = Chem.MolToSmiles(romol)
        mol = Chem.MolFromSmiles(smi)
    else:
        mol = None
    return (flag, None) if mol is None else romol


def _helper2_gen_smiles_to_canon2(smi):
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        res = double_check_valid(m)
        return None if isinstance(res, tuple) else Chem.MolToSmiles(res)
    else:
        return None


def gen_smiles_to_canon2(gen_smis_list, max_n_cpu=30, parallel=True):
    disable_rdkit_logging()

    if parallel:
        cans = mapper(_helper2_gen_smiles_to_canon2, gen_smis_list, max_n_cpu=max_n_cpu)
    else:
        cans = [_helper2_gen_smiles_to_canon2(s) for s in gen_smis_list]
    cans = list(set(filter(None, cans)))
    return cans


def molfromsmiles(smi):
    """For parallelization only"""
    return Chem.MolFromSmiles(smi)


def get_mw(rdmol):
    return Descriptors.MolWt(rdmol)


def get_nHVatoms(rdmol):
    return rdmol.GetNumHeavyAtoms()



