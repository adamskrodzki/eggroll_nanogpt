# nanoGPT + EGGROLL

Based on Evolution Strategies at the Hyperscale [paper](https://arxiv.org/abs/2511.16652)

## Quick start

Prepare the character-level Shakespeare dataset:

```sh
python data/shakespeare_char/prepare.py
```

Train:

```sh
python train.py config/train_shakespeare_char.py
```

Sample from a trained model:

```sh
python sample.py --out_dir=out-shakespeare-char
```

## Strategy

Population sampling strategy can be selected via `strategy=`:
- `standard` — Faithful to original paper
- `standard` — random A/B each generation (default)
- `elitist` — inherits 10% best + children with noise from prior generation
- `greedy`, `greedy_local` — gradient-free variants

## Requirements

```
pip install torch numpy
```

## Credits

Based on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the simplest, fastest repository for training medium-sized GPTs.
