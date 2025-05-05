# Latent Manifold Reconstruction and Representation with Topological and Geometrical Regularization

![show](https://github.com/user-attachments/assets/4d49d799-08f8-4488-8552-5644ba8f30e9)

Accepted by International Conference On Intelligent Computing (ICIC) 2025.

Please access the arXiv pdf version here: 

## Installation
To set up the environment, run the following command in a Terminal:

```
conda env create --file=environment.yml
conda activate mrtg
```

## Demos
We currently have a demo jupyter notebook with the Mammoth dataset. Please run the notebook to get an experience of our model.

## Sources
We use [Pytorch-topological](https://github.com/aidos-lab/pytorch-topological) to realize our topological and geometric model components.

The Mammoth dataset is retrieved from https://github.com/PAIR-code/understanding-umap/blob/master/raw_data/mammoth_3d.json.

The PartNet dataset is retrived from https://huggingface.co/datasets/ShapeNet/PartNet-archive.

The Swiss roll dataset is generated with [Scikit-learn](https://scikit-learn.org/), and the Spheres dataset is generated with Pytorch-topological.
