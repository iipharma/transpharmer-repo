# Transpharmer for novel active ligands

---

TransPharmer is an innovative generative model that integrates ligand-based interpretable pharmacophore fingerprints with generative pre-training transformer (GPT) for de novo molecule generation.
<div align=center>
<img src="demo.jpeg" width="800px">
</div>

A structure of Aspirin is converted into a pharmacophoric topology graph with the shortest topological distance between each feature pair computed. All the two-point and three-point pharmacophoric subgraphs are enumerated, and the topological distances are discretized with specific distance bins. 72- and 108-bit pharmacophore fingerprints are constructed from the two-point pharmacophores with different discretization schemes, while 1032-bit pharmacophore fingerprints are the concatenation of fingerprints of two-point and three-point pharmacophores. The bottom part illustrates the architecture of TransPharmer as a pharmacophore fingerprints-driven GPT decoder.

## Requirements

- inops==0.6.0
- fvcore==0.1.5.post20221221
- guacamol==0.5.2
- numpy==1.23.4
- pandas==1.5.2
- rdkit==2022.9.3
- torch==1.13.1

## Training
To train your own model from command line
```
python train.py --config configs/train.yaml
```

## Benchmarking
To benchmark our model(no-condition version) with guacamol from command line.
```
python benchmark.py --config configs/benchmark.yaml(your config file)
```

## Generation
To generate molecules from command line:
```
python generate.py --config configs/generate_pc.yaml(your config file)
```
A demo case is provided in the [Tutorial](tutorial.ipynb)

## Configurations
all configurations are provided in corresponding *.yaml files:

- benchmark.yaml (configuration of benchmark test )
- generate_nc.yaml (configuration of unconditional generation)
- generate_pc.yaml (configuration of pharmacophore conditional generation)
- train.yaml (configuration of training)

Each yaml file contains model and task-sepcific variables. Details please see [Tutorial](./tutorial.ipynb)  


## Contact

- Youjun Xu (xuyj@iipharma.cn)
- Weixin Xie (1801111477@pku.edu.cn)
- Jianhang Zhang (zhangjh@iipharma.cn)

## License

MIT. See [LICENSE](./LICENSE) for more details.