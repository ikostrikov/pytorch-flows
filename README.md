# pytorch-flows

A PyTorch implementations of [Masked Autoregressive Flow](https://arxiv.org/abs/1705.07057) and 
some other invertible transformations from [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/pdf/1807.03039.pdf) and [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803).

For MAF, I'm getting results similar to ones reported in the paper. GLOW requires some work.

## Run

```bash
python main.py --dataset POWER
```

Available datasets are POWER, GAS, HEPMASS, MINIBONE and BSDS300. For the moment, I removed MNIST and CIFAR10 because I have plans to add pixel-based models later.

## Datasets

The datasets are taken from the [original MAF repository](https://github.com/gpapamak/maf#how-to-get-the-datasets). Follow the [instructions](https://github.com/gpapamak/maf#how-to-get-the-datasets) to get them.

## Tests

Tests check invertibility, you can run them as

```bash
pytest flow_test.py
```