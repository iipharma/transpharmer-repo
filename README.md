# Transpharmer
TransPharmer is an innovative generative model that integrates ligand-based interpretable pharmacophore fingerprints with generative pre-training transformer (GPT) for de novo molecule generation.
<div align=center>
<img src="demo.jpeg" width="800px">
</div>

The chemical structure of Aspirin is converted into a phar- macophoric topology graph with the shortest topological distance between each feature pair computed. All the two-point and three-point pharmacophoric subgraphs are enumerated, and the topological distances are discretized with specific distance bins. 72- and 108-bit pharmacophore fingerprints are constructed from the two-point pharmacophores with different discretization schemes, while 1032-bit pharmacophore finger- prints are the concatenation of fingerprints of two-point and three-point pharmacophores. The right segment illustrates the architecture of TransPharmer as a pharmacophore fingerprints-driven GPT decoder.

## Requirements
```
einops==0.6.0
fvcore==0.1.5.post20221221
guacamol==0.5.2
numpy==1.23.4
pandas==1.5.2
rdkit==2022.9.3
torch==1.13.1
```

## Training
To train your own model from command line
```
python train.py --config configs/train.yaml(your config file)
```

## Benchmarking
To benchmark our model(no-condition version) with guacamol from command line.
```
python benchmark.py --config configs/benchmark.yaml(your config file)
```

## Generate
To generate molecules from command line:
```
python generate.py --config configs/generate_pc.yaml(your config file)
```
We alse provided a toturial based on 2yac ligand for molecule generation see:
[tutorial.ipynb](tutorial.ipynb)

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

## Contact

## Authors and acknowledgment

## MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.