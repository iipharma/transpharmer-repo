# Transpharmer: Accelerating Discovery of Novel and Bioactive Ligands With Pharmacophore-Informed Generative Models

---

TransPharmer is an innovative generative model that integrates interpretable topological pharmacophore fingerprints with generative pre-training transformer (GPT) for de novo molecule generation. TransPharmer can be used to explore pharmacophorically similar and structurally diverse ligands and **has been successfully applied to design novel kinase inhibitors with low-nanomolar potency and high selectivity**. The workflow of TransPharmer is illustrated in the figure below.

For more details, please refer to [our paper](https://www.nature.com/articles/s41467-025-56349-0) in Nature Communications.

If you find TransPharmer useful, please consider citing us as:
```
@article{xie2025accelerating,
  title={Accelerating discovery of bioactive ligands with pharmacophore-informed generative models},
  author={Xie, Weixin and Zhang, Jianhang and Xie, Qin and Gong, Chaojun and Ren, Yuhao and Xie, Jin and Sun, Qi and Xu, Youjun and Lai, Luhua and Pei, Jianfeng},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={2391},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

<div align=center>
<img src="demo.jpeg" width="800px">
</div>

## Installation

TransPharmer was tested on the environment with the following installed packages or configurations.

- python 3.9
- torch=1.13.1
- cuda 11.7 (Nvidia GeForce RTX 3090. GPU memory size: 24 GB)
- rdkit=2022.9.3
- scipy=1.8.0
- numpy=1.23.5
- einops==0.6.0
- fvcore==0.1.5.post20221221
- guacamol=0.5.2 (optional)
- tensorflow=2.11.0 (required by and compatible with guacamol 0.5.2)

Here is the step-by-step process to reproduce TransPharmer's working environment:

1. Create conda environment and activate:

```shell
conda create -n transpharmer python=3.9
```

2. Install pytorch:

```
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. Install other requirements using ``mamba``: first install mamba following the [tutorial](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html#fresh-install-recommended). (Do not forget to change channels as [required](https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html#using-the-defaults-channels).)

```shell
mamba env update -n transpharmer --file other_requirements.yml
```

The reason to use this hybrid installation process is that ``conda`` can be rather annoying and time-consuming to solve environment and settle all requirements. ``mamba`` is faster.

4. (Optional) run GuacaMol benchmarking: need to adjust some packages in order to be compatible with GuacaMol. (GuacaMol will automatically install the latest version of TensorFlow, which can be problematic.)
```
Manually upgrade or downgrade the following packages with pip for compatibility:
- tensorflow=2.11.0
- scipy=1.8.0
- numpy=1.23.5
```

## Download data and model weights
We used the GuacaMol pre-built datasets to train and validate our model, which can be downloaded [here](https://github.com/BenevolentAI/guacamol?tab=readme-ov-file#download). (However, if the links are inaccessible temporarily, we also provide a copy [here](https://disk.pku.edu.cn/link/AAB1F5B8730D4B4522ABCD447BFD74A23E), since GuacaMol is under MIT license.)

```shell
cd transpharmer-repo/
unzip guacamol.zip
ls data/
```

Pretrained TransPharmer model weights can be downloaded [here](https://disk.pku.edu.cn/link/AAE3AFB88B939E4C1CB7A6EF0C7311716F). The organization of the downloaded directory is described as follows:
```
cd transpharmer-repo/
unzip TransPharmer_weights.zip
ls weights/

weights/
  guacamol_pc_72bit.pt: trained with 72-bit pharmacophore fingerprints of GuacaMol compounds;
  guacamol_pc_80bit.pt: trained with 72-bit pharmacophore fingerprints concated with feature count vectors;
  guacamol_pc_108bit.pt: trained with 108-bit pharmacophore fingerprints;
  guacamol_pc_1032bit.pt: trained with 1032-bit pharmacophore fingerprints;
  guacamol_nc.pt: unconditional version trained on GuacaMol;
  moses_nc.pt: unconditional version trained on MOSES;
```


## Configurations
We provide several configuration files (*.yaml) for different utilities:

- generate_pc.yaml (configuration of **p**harmacophore-**c**onditioned generation)
- generate_nc.yaml (configuration of unconditional/**n**o-**c**ondition generation)
- benchmark.yaml (configuration of benchmarking)
- train.yaml (configuration of training)

Each yaml file contains model and task-specific parameters. See `explanation.yaml` and [Tutorial](./tutorial.ipynb) for more details.

## Training
To train your own model, use the following command line:
```
python train.py --config configs/train.yaml
```

## Generation
To generate molecules similar to input reference compounds in terms of pharmacophore, usually some known actives, use the following command line:
```
python generate.py --config configs/generate_pc.yaml
```

Generated SMILESs are saved in the user-specified csv file. The generated csv file has two columns (`Template` and `SMILES`) in pharmacophore-conditioned (pc) generation mode (or one column `SMILES` in unconditional (nc) generation mode).

A demo is provided in the [Tutorial](tutorial.ipynb).

## Benchmarking
To benchmark our unconditional model with GuacaMol, run the following command line:
```
python benchmark.py --config configs/benchmark.yaml
```

To benchmark our unconditional model with MOSES, generate samples using `benchmark.py` or `generate.py` with `generate_nc.yaml` config and compute all MOSES metrics following their [guidelines](https://github.com/molecularsets/moses?tab=readme-ov-file#benchmarking-your-models).

To evaluate $D_{\rm count}$ and $S_{\rm pharma}$ metrics (see definition in our paper) for pharmacophore-based generation, run `get_metrics` in `benchmark.py` with generated csv file.

## Reproduction of PLK1 inhibitors case study
To reproduce the results of case study of designing PLK1 inhibitors in our paper, run the following command:

```shell
python generate.py --config configs/generate_reproduce_plk1.yaml
```

After filtering out invalid and duplicate SMILES, one should be able to find exactly the same structure as `lig-182` (corresponds to the most potent synthesized compound `IIP0943`) and structures with identical BM scaffolds to `lig-3`, `lig-524` and `lig-886`.

## Contact

- Youjun Xu (xuyj@iipharma.cn)
- Weixin Xie (xiewx@pku.edu.cn)
- Jianhang Zhang (zhangjh@iipharma.cn)

## License

MIT license. See [LICENSE](./LICENSE) for more details.

## Acknowledgement

We thank authors of the following repositories to share their codes:

- https://github.com/rdkit/rdkit
- https://github.com/BenevolentAI/guacamol
- https://github.com/devalab/molgpt
