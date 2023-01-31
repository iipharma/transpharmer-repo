from rdkit import Chem
from .pharm.misc_v2 import get_pharma_fp, get_pharma_count_fp
import time
import random
import rdkit.RDLogger as rkl
rdlogger = rkl.logger()
rdlogger.setLevel(rkl.CRITICAL)
from .io import load_from_csv


def get_props(csv_fn):
    props = set()
    mols = load_from_csv(csv_fn)
    for mol in mols:
        prop = get_pharma_fp(mol)
        prop = ''.join([str(t) for t in prop])
        props.add(prop)
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


def get_props_from_smiles(smiles_list, num_props):
    props = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            if num_props == 72:
                prop = get_pharma_fp(mol)
            elif num_props == 80:
                prop = get_pharma_count_fp(mol)
            else:
                raise ValueError
            prop = [t for t in prop]
            if prop not in props:
                props.append(prop)
    return props


def cleanise_smiles(smiles, sanitize_mol=True):
    '''a function to cleanise smiles into a starndard kekulized smiles string'''
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