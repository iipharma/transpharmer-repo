import re, math
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from utils.pharm.misc_v2 import get_pharma_count, get_pharma_fp


pad_token = '<'
nan_token = ''
tokens = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', pad_token, '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]',
            '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']
vocab_size = len(tokens)
stoi = { ch:i for i,ch in enumerate(tokens) }
itos = { i:ch for i,ch in enumerate(tokens) }

pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)


def tokenize(smiles, max_len=0):
    tokens = [stoi[s] for s in regex.findall(smiles)]
    if max_len > 0:
        tokens += [stoi[pad_token]]*(max_len-len(tokens))
        tokens = tokens[:max_len]
    return tokens


def untokenize(tokens):
    smiles = ''.join([itos[int(t)] for t in tokens])
    smiles = smiles.replace(pad_token, nan_token)
    return smiles


class SmileDataset(Dataset):
    def __init__(self, args, dataframe, block_size, num_props, aug_prob=0.5, isomericSmiles=True, canonical=False):
        self.args = args
        self.num_props = num_props
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = dataframe
        self.aug_prob = aug_prob
        self.isomericSmiles = isomericSmiles
        self.canonical = canonical
        self.buffer = None
    
    def __len__(self):
        return len(self.data)

    def get_prop(self, mol):
        try:
            ct = get_pharma_count(mol)
            fp = get_pharma_fp(mol)
            fp = [t for t in fp]
            prop = ct + fp
            return prop
        except:
            return

    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def randomize(self, smi):
        try:
            if np.random.uniform() < self.aug_prob:
                smi = self.randomize_smiles(smi)
            dix = tokenize(smi, self.max_len)
            return dix
        except:
            return

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        smi = data.smiles.strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            dix = self.randomize(smi)
            if dix is not None:
                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)
                if self.num_props:
                    prop = self.get_prop(mol)
                    if prop is not None:
                        p = torch.tensor(prop, dtype = torch.float)
                        if self.buffer is None:
                            self.buffer = (x, y, p)
                        return x, y, p
                else:
                    if self.buffer is None:
                        self.buffer = (x, y)
                    return x, y
        return self.buffer
    

def build_dataloader(dataset, is_train, batch_size, num_workers):
    dataloader = DataLoader(dataset, 
                            shuffle=is_train,
                            pin_memory=True,
                            batch_size=batch_size,
                            num_workers=num_workers)
    return dataloader