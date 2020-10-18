# Checking the effectiveness of Domain Adversarial Neural Networks on Kaggle competition(s)

[Domain Adversarial Neural Networks](https://arxiv.org/abs/1505.07818) is a famous Unsupervised Domain Adaptation (UDA) algorithm and the first to combine Representation Learning with Domain Adaptation. It is said to be effective in the case where Source and Target follows the same input - output relationship whereas the distribution of the input is different between Source and Target (the case known as *Covariate Shift*).

In kaggle competition, it is often the case that a large discrepancy is present between train dataset and test dataset(domain shift). In some case, domain adaptation algorithms like DANN seem to fit for the task. However, to the best of our knowledge, these algorithms are seldom used in kaggle competitions. This may be because

1. Kagglers don't know these algorithms. The ideas are not imported to the community.
2. These algorithms are not useful for those competitions. The capabilities of those algorithms claimed in the paper have not been demonstrated in those competitions.

The purpose of this repository is to check whether DANN is to verify that DANN is actually an algorithm that can be used in Kaggle competitions.

The result is reported in this [blog post](https://zenn.dev/koukyo1994/articles/8ebac81fd74d2f4f0905) (Japanese).

## Installation

`Python >= 3.6` is required. Hardware with GPU accelerator is recommended.

```shell
pip install -r requirements.txt
```

## How to reproduce

### MNIST vs MNISTM

1. Download MNISTM dataset from [Google Drive](https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg) and put in `input/digits/`.
2. Run the command below.

```shell
python main.py --config configs/000_digits.yml
```

To run training without DANN:

```shell
python main.py --config configs/002_digits_naive.yml
```

To get the figures in `assets/digits/`, first run training with the two commands above and then run

```shell
python digitviz.py
```

## References

\[1\]: [fungtion/DANN](https://github.com/fungtion/DANN)

\[2\]: [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
