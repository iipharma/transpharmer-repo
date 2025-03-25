# misc func
# Create by xwx 2022-2-7
import os
from rdkit import Chem, DataStructs
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem import ChemicalFeatures, AllChem, rdReducedGraphs
from rdkit.Chem.SaltRemover import SaltRemover
from enum import Enum
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import numpy as np


BaseFeatures_path = f"{os.path.dirname(__file__)}/BaseFeatures.fdef"


class Pharmacophores(Enum):
    Donor = 'Donor'
    Acceptor ='Acceptor'
    NegIon = 'NegIonizable'
    PosIon = 'PosIonizable'
    ZnB = 'ZnBinder'
    Aromatic = 'Aromatic'
    Hydro = 'Hydrophobe'
    LumHydro = 'LumpedHydrophobe'


AllPharmaTypes = [x.value for x in Pharmacophores]


def get_pharma_fp(romol, n_dim=72, return_list=True):

    if n_dim == 72:
        minPointCount = 2
        maxPointCount = 2
        n_dist_bins = 2
    elif n_dim == 108:
        minPointCount = 2
        maxPointCount = 2
        n_dist_bins = 3
    elif n_dim == 1032:
        minPointCount = 2
        maxPointCount = 3
        n_dist_bins = 2
    else:
        raise ValueError(f"Invalid argument n_dim={n_dim}")

    fdefName = BaseFeatures_path
    fdef = ChemicalFeatures.BuildFeatureFactory(fdefName)

    sigFactory = SigFactory(fdef, minPointCount=minPointCount, maxPointCount=maxPointCount)
    if n_dist_bins == 2:
        sigFactory.SetBins([(0, 3), (3, 8)])  # 0 < d < 3 距离index为0; 3 <= d < 8 距离index为1
    elif n_dist_bins == 3:
        sigFactory.SetBins([(0, 2), (2, 5), (5, 8)])
    else:
        raise NotImplementedError()
    sigFactory.Init()

    fp = Generate.Gen2DFingerprint(romol, sigFactory, dMat=None)
    if return_list:
        fp = [b for b in fp]
        return fp
    else:
        return fp


def get_pharma_count(romol, cid=-1):
    """
    """
    #####################
    # pharma count generation
    #####################
    fdef = ChemicalFeatures.BuildFeatureFactory(BaseFeatures_path)
    feats = fdef.GetFeaturesForMol(romol, confId=cid)
    count = dict([(type_, 0) for type_ in AllPharmaTypes])
    for f in feats:
        count[f.GetFamily()] += 1
    count = list(count.values())
    return count


def get_pharma_count_fp(romol):
    """
    """
    #####################
    # pharma count generation
    #####################
    count = get_pharma_count(romol)
    fp = get_pharma_fp(romol, n_dim=72)
    prop = count + fp
    return prop


def get_sigFactory(minPointCount=2, maxPointCount=2, n_dist_bins=2,
                    which_factory="sig"):
    """
    """
    # fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    fdefName = BaseFeatures_path
    fdef = ChemicalFeatures.BuildFeatureFactory(fdefName)

    sigFactory = SigFactory(fdef, minPointCount=minPointCount, maxPointCount=maxPointCount)
    if n_dist_bins == 2:
        sigFactory.SetBins([(0, 3), (3, 8)])
    elif n_dist_bins == 3:
        sigFactory.SetBins([(0, 2), (2, 5), (5, 8)])
    else:
        raise NotImplementedError()
    sigFactory.Init()
    # return sigFactory if which_factory == "sig" else fdef
    return sigFactory, fdef


def _mapper(n_jobs):
    if n_jobs == 1:
        def __mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return __mapper
    elif isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def __mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()  # if exception raised, terminate all process before exit
            return result

        return __mapper
    else:
        return n_jobs.map


def mapper(func, iterable, max_n_cpu=10, desc=None, verbose=True):
    n_cpu = min(max_n_cpu, len(iterable))
    if verbose:
        return _mapper(n_cpu)(func, tqdm(iterable, desc=desc))
    else:
        return _mapper(n_cpu)(func, iterable)


def MolRemoveIsotopes(romol):
    """去掉同位素信息：会改变输入的分子
    Adapted from https://sourceforge.net/p/rdkit/mailman/message/36877847/
    """
    atom_data = [(atom, atom.GetIsotope()) for atom in romol.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            atom.SetIsotope(0)


def neutralize_atoms(romol):
    """in-place op"""
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = romol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = romol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()


def remove_salt(romol):
    if "." in Chem.MolToSmiles(romol):
        remover = SaltRemover()
        m = remover.StripMol(romol, dontRemoveEverything=True)
        if "." in Chem.MolToSmiles(m):  # case: contain salts out of filters
            return None
        else:
            return m
    else:
        return romol


def cleanse_mol(romol):
    """"""
    op_mol = Chem.Mol(romol)  # a good copy

    # 去掉同位素信息
    MolRemoveIsotopes(op_mol)
    # 中和电荷
    try:
        neutralize_atoms(op_mol)
    except Chem.rdchem.AtomValenceException:  # 可能会有rdkit.Chem.rdchem.AtomValenceException等错误，一律丢掉
        return None
    # 去盐
    op_mol = remove_salt(op_mol)
    return op_mol


def get_fp_for_sim_calc(mol, fptype,
                        radius=2, nBits=2048,
                        ph4_dim=72,
                        input_smiles=False):
    """获取指纹用于计算分子相似度
    为了获取两个分子的最大可能的相似度，先经过清洗
    """
    if input_smiles:
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None

    mol = cleanse_mol(mol)
    if mol:
        if fptype == "morgan":  # UIntSparseIntVect类型，长度自动指定，非常非常长，而且记录子结构出现次数；但不同分子的指纹长度相同
            # 转为np.ndarray的消耗非常大; count-based
            return AllChem.GetMorganFingerprint(mol, radius)
        elif fptype == "morgan-bv":  # 没法与上面的"morgan"指纹对象直接计算相似度
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)  # 长度由nBits指定，记录有无出现某种子结构
        elif fptype == "rdkit_pharm":
            return get_pharma_fp(mol, n_dim=ph4_dim, return_list=False)
        elif fptype == "erg":  # ErG fingerprint: 315 positive real number 可以大于1
            return rdReducedGraphs.GetErGFingerprint(mol).tolist()  # 原本输出为np.ndarray，但是很不方便，特别是评估`None in [fp1, fp2]`
        else:
            raise NotImplementedError()
    else:
        return None


def ergfp_tanimoto(fp1, fp2):
    """Follow https://iwatobipen.wordpress.com/2016/01/16/ergfingerprint-in-rdkit/"""
    fp1 = np.array(fp1)
    fp2 = np.array(fp2)
    denom = (np.sum(np.dot(fp1, fp1)) + np.sum(np.dot(fp2, fp2)) - np.sum(np.dot(fp1, fp2)))
    num = np.sum(np.dot(fp1, fp2))
    if denom == 0 and num == 0:
        return 0.0
    else:
        return num / denom


def bulk_fp_similarity(ref_mol, query_mols, input_smiles=False,
                       fp_type="morgan",
                       morgan_radius=2, nBits=2048,
                       ph4_dim=72,
                       parallel=True):
    """一个参考分子，多个待评估分子之间的相似度
    Args:
        * parallel: 内部并行。当外部并行调用bulk_fp_similarity时，设置为False
        * input_smiles: query_mols是否为smiles列；用于计算上百万分子时，比先生成rdkit mol对象极为节省内存

    Note: 需要对分子进行清洗，不要考虑分子状态的对相似度影响
    """
    ref_fp = get_fp_for_sim_calc(ref_mol, fptype=fp_type, radius=morgan_radius, nBits=nBits,
                                 ph4_dim=ph4_dim)

    p_func = partial(get_fp_for_sim_calc, fptype=fp_type, radius=morgan_radius, nBits=nBits,
                     ph4_dim=ph4_dim, input_smiles=input_smiles)
    if parallel:
        query_fps = mapper(p_func, query_mols, desc="[bulk_fp_similarity]")
    else:
        query_fps = [p_func(m) for m in query_mols]
    if None in query_fps:
        # mask = mapper(n_cpus)(_helper2, query_fps)
        mask = [True if fp else False for fp in query_fps]  # 没问题的分子/指纹标记为True
        query_fps = list(filter(None, query_fps))
    else:
        mask = None

    if fp_type == "erg":
        pf = partial(ergfp_tanimoto, fp2=ref_fp)
        if parallel:
            similarity = mapper(pf, query_fps)
        else:
            similarity = [pf(fp) for fp in query_fps]
    else:
        similarity = DataStructs.BulkTanimotoSimilarity(ref_fp, query_fps)

    if mask:
        return similarity, mask
    else:
        return similarity


def get_D_count(ref_m, gen_m):
    """Get D_count used in benchmarking pharmacophore-based molecular generative models"""
    return sum(abs(e1-e2) for e1, e2 in zip(get_pharma_count(gen_m), get_pharma_count(ref_m)))


def get_S_pharma(ref_m, gen_ms):
    """Get S_pharma used in benchmarking pharmacophore-based molecular generative models"""
    return bulk_fp_similarity(ref_m, gen_ms, fp_type="erg")

