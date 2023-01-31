# Transpharmer

## Introduction
Codes for our paper "Explore New Focused Chemical Space using Fuzzy Pharmacophore
Fingerprint Driven Transformer Decoders". We can generate molecules with supplied pharmacophore fingerprints. Search space can be further contrained with specific scaffold.

## Requirements 
    einops==0.6.0
    fvcore==0.1.5.post20221221
    guacamol==0.5.2
    numpy==1.23.4
    pandas==1.5.2
    rdkit==2022.9.3
    torch==1.13.1

## Training
To train your own model from command line
```
python train.py
```

## Benchmarking
To benchmark our model(no-condition version) with guacamol from command line.
```
python benchmark.py
```

## Generate
To generate molecules from command line:
```
python generate.py
```
We alse provided a toturial based on 2yac ligand for molecule generation see:
```
tutorial.ipynb
```

## Configurations
all configurations are provided in corresponding .yaml files:
```
benchmark.yaml
generate_nc.yaml
generate_pc.yaml
train.yaml
```
each file contains model and task-sepcific variables

## Changelog
```
230131 init commit
```

### Contact

## Authors and acknowledgment

## License