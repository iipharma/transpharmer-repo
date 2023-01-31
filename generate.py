"""
Modified from generate.py by xuyj
"""
import torch
from model import GPT, sample
from dataset import tokenize, untokenize
from utils.io import load_config
from utils.mol import get_props_from_smiles, cleanise_smiles
import typing as t
import logging
logging.getLogger().setLevel(logging.INFO)


def generate(config):
    model = GPT(config)
    model.load_state_dict(torch.load(config.GENERATE.CKPT_PATH))
    model = model.to(config.DEVICE)
    context = torch.tensor(tokenize(config.GENERATE.CONTEXT), dtype=torch.long)[None, ...].to(config.DEVICE)
    if config.MODEL.NUM_PROPS:
        props = get_props_from_smiles(config.GENERATE.TEMPLATES, config.MODEL.NUM_PROPS)
        props = [torch.tensor(prop, dtype=torch.int).to(config.DEVICE) for prop in props]
    else:
        props = [None]
    gen_smiles = set()
    num_repeat, num_invalid, num_total = 0, 0, 0

    while(True):
        for prop in props:
            c = context.repeat(config.GENERATE.BATCH_SIZE, 1)
            if prop is not None:
                p = prop.repeat(config.GENERATE.BATCH_SIZE, 1)
            else:
                p = None
            gen_tokens: t.List[t.List[int]] = sample(model=model, x=c, steps=config.MODEL.MAX_LEN, temperature=config.GENERATE.TEMPERATURE, sample=True, top_k=None, prop=p)
            del c, p
            for tokens in gen_tokens:
                smiles: str = untokenize(tokens)
                num_total +=1
                new_smiles = cleanise_smiles(smiles, sanitize_mol=True)
                if new_smiles is not None:
                    len0 = len(gen_smiles)
                    gen_smiles.add(new_smiles)
                    len1 = len(gen_smiles)
                    if len1 - len0 ==1:
                        with open(config.GENERATE.OUTPUT, 'a') as f:
                            f.write(f'{smiles}\n')
                        if len1 % 100 ==0:
                            logging.info(f"Statictic: unique:{float(len1)/num_total} | repeat:{float(num_repeat)/num_total} | invalid:{float(num_invalid)/num_total} | total:{num_total}")
                        if len1 > config.GENERATE.TOTAL:
                            return
                    else:
                        num_repeat +=1
                else:
                    num_invalid +=1


if __name__ == '__main__':
    config = load_config('configs/generate_pc.yaml')
    generate(config)