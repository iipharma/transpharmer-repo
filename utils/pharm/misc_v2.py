# misc func
# Create by xwx 2022-2-7
import os
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem import ChemicalFeatures
from enum import Enum


BaseFeatures_path = f"{os.path.dirname(__file__)}/BaseFeatures.fdef"
minPointCount=2
maxPointCount=2
n_dist_bins=2

fdef = ChemicalFeatures.BuildFeatureFactory(BaseFeatures_path)
sigFactory = SigFactory(fdef, minPointCount=minPointCount, maxPointCount=maxPointCount)
if n_dist_bins == 2:
    sigFactory.SetBins([(0, 3), (3, 8)])
elif n_dist_bins == 3:
    sigFactory.SetBins([(0, 2), (2, 5), (5, 8)])
else:
    raise NotImplementedError()
sigFactory.Init()


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


def get_pharma_fp(romol):
    """
    """
    #####################
    # fingerprint generation
    #####################
    fp = Generate.Gen2DFingerprint(romol, sigFactory, dMat=None)
    fp = [t for t in fp]
    return fp


def get_pharma_count(romol, cid=-1):
    """
    """
    #####################
    # pharma count generation
    #####################
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
    fp = get_pharma_fp(romol)
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