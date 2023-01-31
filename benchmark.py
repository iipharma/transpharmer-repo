# benchmarking Transpharmer model(no condition version) with guacamol api
from typing import List
import random
import torch
import json
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import assess_distribution_learning
from model import GPT, sample
from dataset import tokenize, untokenize
from utils.io import load_config
from datetime import date
import typing as t
from pathlib import Path


class TransPharmerGenerator(DistributionMatchingGenerator):
    def __init__(self, config):
        """
        Args:
            * baseline_mode: Sample from test-set directly
            * batch_size: sample batch size
            * n_repeat: number of times each condition grow
        """
        self.config = config
        if self.config.BENCHMARK.GEN_SMILES_PATH:
            assert not Path(self.config.BENCHMARK.GEN_SMILES_PATH).exists(), "Existed file to save generated smiles!"
        self.model = GPT(config)
        self.model.load_state_dict(torch.load(config.BENCHMARK.CKPT_PATH))
        self.model.to(config.DEVICE)

    def generate(self, number_samples: int) -> List[str]:
        gen_smiles = []
        if not self.config.BENCHMARK.BASELINE_MODE:
            context = torch.tensor(tokenize(self.config.BENCHMARK.CONTEXT), dtype=torch.long)[None, ...].to(self.config.DEVICE)
            while len(gen_smiles) <= number_samples:
                c = context.repeat(self.config.BENCHMARK.BATCH_SIZE, 1)
                gen_tokens: t.List[t.List[int]] = sample(model=self.model, x=c, steps=self.config.MODEL.MAX_LEN, temperature=self.config.BENCHMARK.TEMPERATURE, sample=True, top_k=self.config.BENCHMARK.TOP_K, prop=None)
                del c
                for tokens in gen_tokens:
                    smiles: str = untokenize(tokens)
                    gen_smiles.append(smiles)
        else:
            with open(self.config.BENCHMARK.CHEMBL_TRAINING_FILE, "r") as f:
                for l in f.readlines():
                    gen_smiles.append(l.strip())
        random.seed(self.config.RANDOM_SEED)
        random.shuffle(gen_smiles)
        return gen_smiles[:number_samples]


def run_assessment(config):
    print(f"Job-{date.today()}")
    config.BENCHMARK.OUTPUT_JSON = str(Path(config.BENCHMARK.RES_DIR) / config.BENCHMARK.OUTPUT_JSON)
    config.BENCHMARK.CHEMBL_TRAINING_FILE = str(Path(config.BENCHMARK.DATA_DIR) / config.BENCHMARK.CHEMBL_TRAINING_FILE)
    generator = TransPharmerGenerator(config=config)
    assess_distribution_learning(generator, chembl_training_file=config.BENCHMARK.CHEMBL_TRAINING_FILE,
                                 json_output_file=config.BENCHMARK.OUTPUT_JSON)
    with open(config.BENCHMARK.OUTPUT_JSON, "r") as f:
        res = json.load(f)
    print([(d_['benchmark_name'], d_['score']) for d_ in res["results"]])


if __name__ == '__main__':
    config = load_config('configs/benchmark.yaml')
    run_assessment(config)