# benchmarking Transpharmer model(no condition version) with guacamol api
from typing import List
import typing as t
import torch
import json
from datetime import date
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from functools import partial
from rdkit import Chem
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import assess_distribution_learning

from model import GPT, sample
from dataset import tokenize, untokenize
from utils.io import load_config
from utils.pharm.misc import mapper, bulk_fp_similarity, get_pharma_count
from utils.mol import gen_smiles_to_canon2, molfromsmiles, get_mw, get_nHVatoms
from utils.seed import set_seed


class TransPharmerGenerator(DistributionMatchingGenerator):
    def __init__(self, config):
        """
        """
        self.config = config
        self.model = GPT(config)
        self.model.load_state_dict(torch.load(config.BENCHMARK.CKPT_PATH))
        self.model.to(config.DEVICE)

        if self.config.BENCHMARK.GEN_SMILES_PATH:
            assert not Path(self.config.BENCHMARK.GEN_SMILES_PATH).exists(), "Existed file to save generated smiles!"

        set_seed(config.RANDOM_SEED)


    def generate(self, number_samples: int) -> List[str]:
        gen_smiles = []
        if not self.config.BENCHMARK.BASELINE_MODE:
            context = torch.tensor(tokenize(self.config.BENCHMARK.CONTEXT), dtype=torch.long)[None, ...].to(self.config.DEVICE)
            while len(gen_smiles) < number_samples:
                c = context.repeat(self.config.BENCHMARK.BATCH_SIZE, 1)
                gen_tokens: t.List[t.List[int]] = sample(model=self.model, x=c, steps=self.config.MODEL.MAX_LEN, temperature=self.config.BENCHMARK.TEMPERATURE, sample_=True, top_k=self.config.BENCHMARK.TOP_K, prop=None)
                del c
                for tokens in gen_tokens:
                    smiles = untokenize(tokens)
                    gen_smiles.append(smiles)
        else:
            with open(self.config.BENCHMARK.CHEMBL_TRAINING_FILE, "r") as f:
                for l in f.readlines():
                    gen_smiles.append(l.strip())
        # random.seed(self.config.RANDOM_SEED)
        # random.shuffle(gen_smiles)

        if self.config.BENCHMARK.GEN_SMILES_PATH:  # for MOSES
            with open(self.config.BENCHMARK.GEN_SMILES_PATH, "w") as fo:
                fo.write("SMILES\n")
                fo.write("\n".join(gen_smiles[:number_samples]))
                fo.write("\n")
            print(f"Saved in {self.config.BENCHMARK.GEN_SMILES_PATH}")
        else:  # for GuacaMol
            return gen_smiles[:number_samples]


def run_assessment(config):
    print(f"Job-{date.today()}")
    # config.BENCHMARK.OUTPUT_JSON = str(Path(config.BENCHMARK.RES_DIR) / config.BENCHMARK.OUTPUT_JSON)
    # config.BENCHMARK.CHEMBL_TRAINING_FILE = str(Path(config.BENCHMARK.DATA_DIR) / config.BENCHMARK.CHEMBL_TRAINING_FILE)
    generator = TransPharmerGenerator(config=config)
    assess_distribution_learning(generator, chembl_training_file=config.BENCHMARK.CHEMBL_TRAINING_FILE,
                                 json_output_file=config.BENCHMARK.OUTPUT_JSON)
    with open(config.BENCHMARK.OUTPUT_JSON, "r") as f:
        res = json.load(f)
    print([(d_['benchmark_name'], d_['score']) for d_ in res["results"]])


def _score_1_batch_helper2(mol, ref_counts):
    return sum(abs(e1-e2) for e1, e2 in zip(get_pharma_count(mol), ref_counts))


def _get_metrics_helper(item):
    ## get mols: 生成分子挑出valid和unique的部分
    _, [ref_smi, gen_smis] = item  # dont care about the name
    ref_m = Chem.MolFromSmiles(ref_smi)
    gen_smis = set(gen_smiles_to_canon2(gen_smis, parallel=False))
    gen_ms = [molfromsmiles(s) for s in gen_smis]
    gen_ms = list(filter(None, gen_ms))

    scores = dict()
    ## compute scores
    # ph4 similarity
    tmp = bulk_fp_similarity(ref_m, gen_ms, fp_type="erg", parallel=False)
    if isinstance(tmp, tuple):
        tmp, mask = tmp
        gen_ms = [m for m, pass_ in zip(gen_ms, mask) if pass_]
    scores["ph4_similarity"] = np.mean(tmp)

    # morgan similarity
    tmp = bulk_fp_similarity(ref_m, gen_ms, fp_type="morgan", parallel=False)
    if isinstance(tmp, tuple):
        tmp, mask = tmp
        gen_ms = [m for m, pass_ in zip(gen_ms, mask) if pass_]
    # scores.append(np.mean(tmp))
    scores["morgan_similarity"] = np.mean(tmp)

    # ph4 feature count deviation V2
    pf = partial(_score_1_batch_helper2, ref_counts=get_pharma_count(ref_m))
    tmp = [pf(m) for m in gen_ms]
    scores["count_deviation_v2"] = np.mean(tmp)  # pool over samples

    # get valid and unique rate
    scores["valid_unique"] = len(gen_ms) / len(item[1][1])

    # get MW deviation
    ref_mw = get_mw(ref_m)
    scores["mw_deviation"] = np.mean([get_mw(m)-ref_mw for m in gen_ms])

    # get number of heavy atoms deviation
    ref_nh = get_nHVatoms(ref_m)
    scores["nhv_deviation"] = np.mean([get_nHVatoms(m) - ref_nh for m in gen_ms])

    return scores


def get_metrics(gen_csv,
                gen_conds_subset=None):
    """Eval D_count, S_pharma and other metrics for pharmacophore-conditioned generation
    """
    ## get organized ref and gen smiles
    data = {i: [t_, list(group['SMILES'])] for i, (t_, group) in enumerate(pd.read_csv(gen_csv).groupby('Template'))}

    if gen_conds_subset:
        data = {k: v for k, v in data.items() if k in gen_conds_subset}

    ## compute scores and statistics
    # only care about statistics over ref mols
    res = mapper(_get_metrics_helper, data.items())
    cols = list(res[0].keys())
    res = np.vstack([list(d.values()) for d in res])

    means = np.mean(res, axis=0)
    stds = np.std(res, axis=0)
    # print(f"Valid and unique ratio within one condition: min-{min(res[:, 3]):.4f}")
    df = pd.DataFrame({col: [f"{m:.3f}±{s:.3f}"] for col, m, s in zip(cols, means, stds)})
    # df["valid_unique_min"] = [f"{min(res[:, 4]):.4f}"]
    return df
    # print(df)



if __name__ == '__main__':

    # run benchmarking
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/benchmark.yaml', help='specify config file')
    args = parser.parse_args()
    config = load_config(args.config)
    # GuacaMol
    run_assessment(config)
    ## MOSES
    # generator = TransPharmerGenerator(config=config)
    # generator.generate(30000)

    # # compute pharmacophore-based generation metrics
    # gen_csv_ = "./output/reproduce_plk1_case/out_test.smi"
    # get_metrics(gen_csv_)


