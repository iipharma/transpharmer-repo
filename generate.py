import torch
import argparse
import logging
logging.getLogger().setLevel(logging.INFO)
from rdkit import Chem
import os

from model import GPT, sample
from dataset import tokenize, untokenize
from utils.seed import set_seed
from utils.io import load_config
from utils.mol import get_props_from_mols
from utils.pharm.misc import mapper


def batchify_a_list(list_, size):
    for i in range(0, len(list_), size):
        yield list_[i:i + size]


def generate(input_config):
    model = GPT(input_config)
    model.load_state_dict(torch.load(input_config.GENERATE.CKPT_PATH))
    model = model.to(input_config.DEVICE)

    if isinstance(input_config.GENERATE.CONTEXT, str):
        all_context = torch.tensor(tokenize(input_config.GENERATE.CONTEXT), dtype=torch.long).repeat(
            input_config.GENERATE.N_REPEAT * len(input_config.GENERATE.TEMPLATES), 1)
    elif isinstance(input_config.GENERATE.CONTEXT, list):
        assert len(input_config.GENERATE.CONTEXT) == len(input_config.GENERATE.TEMPLATES), \
            "Unmatched lengths of contexts and templates"
        all_context = torch.tensor([tokenize(context) for context in input_config.GENERATE.CONTEXT],
                                   dtype=torch.long).repeat_interleave(input_config.GENERATE.N_REPEAT, dim=0)
    else:
        raise ValueError()

    if input_config.MODEL.NUM_PROPS:
        mols = [Chem.MolFromSmiles(s) for s in input_config.GENERATE.TEMPLATES]
        assert None not in mols, f"Found invalid smiles in TEMPLATES: {input_config.GENERATE.TEMPLATES[mols.index(None)]}"
        props = get_props_from_mols(mols, input_config.MODEL.NUM_PROPS)
        all_props = torch.repeat_interleave(torch.tensor(props, dtype=torch.int), input_config.GENERATE.N_REPEAT, dim=0)

        all_templates = []
        for t_ in input_config.GENERATE.TEMPLATES:
            all_templates.extend([t_] * input_config.GENERATE.N_REPEAT)
        with open(input_config.GENERATE.OUTPUT, "w") as fo:
            fo.write("Template,SMILES\n")  # write csv header
    else:
        all_props = None
        with open(input_config.GENERATE.OUTPUT, "w") as fo:
            fo.write("SMILES\n")  # write csv header

    set_seed(input_config.RANDOM_SEED)
    n_gens = 0
    if not os.path.exists(os.path.dirname(input_config.GENERATE.OUTPUT)):
        os.makedirs(os.path.dirname(input_config.GENERATE.OUTPUT))
    if os.path.exists(input_config.GENERATE.OUTPUT):
        logging.warning(f"Output file {input_config.GENERATE.OUTPUT} already exists. "
                        f"Appending to the existing file.")

    if input_config.MODEL.NUM_PROPS:
        for context, prop, templates in zip(torch.split(all_context, input_config.GENERATE.BATCH_SIZE),
                                            torch.split(all_props, input_config.GENERATE.BATCH_SIZE),
                                            batchify_a_list(all_templates, input_config.GENERATE.BATCH_SIZE)):
            context = context.to(input_config.DEVICE)
            prop = prop.to(input_config.DEVICE)

            gen_tokens = sample(model=model, x=context, steps=input_config.MODEL.MAX_LEN,
                                temperature=input_config.GENERATE.TEMPERATURE, sample_=True, top_k=None, prop=prop)
            gen_tokens = gen_tokens.cpu().numpy().tolist()
            gen_smiles = mapper(untokenize, gen_tokens, max_n_cpu=input_config.GENERATE.N_CPU, verbose=False)

            lines = [f"{t_},{raw_smi}\n" for t_, raw_smi in zip(templates, gen_smiles)]
            with open(input_config.GENERATE.OUTPUT, "a") as fo:
                fo.write("".join(lines))

            n_gens += len(gen_smiles)
            if n_gens % 1024 == 0:
                logging.info(f"Number of sampled SMILES: {n_gens}")

    else:
        for context in torch.split(all_context, input_config.GENERATE.BATCH_SIZE):
            context = context.to(input_config.DEVICE)

            gen_tokens = sample(model=model, x=context, steps=input_config.MODEL.MAX_LEN,
                                temperature=input_config.GENERATE.TEMPERATURE, sample_=True, top_k=None, prop=None)
            gen_tokens = gen_tokens.cpu().numpy().tolist()
            gen_smiles = mapper(untokenize, gen_tokens, max_n_cpu=input_config.GENERATE.N_CPU, verbose=False)

            lines = [f"{raw_smi}\n" for t_, raw_smi in gen_smiles]
            with open(input_config.GENERATE.OUTPUT, "a") as fo:
                fo.write("".join(lines))

            n_gens += len(gen_smiles)
            if n_gens % 1024 == 0:
                logging.info(f"Number of sampled SMILES: {n_gens}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/generate_pc.yaml', help='specify config file')
    args = parser.parse_args()

    config = load_config(args.config)
    generate(config)

