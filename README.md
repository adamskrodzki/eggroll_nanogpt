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
- `standard` — Faithful to original paper (best val loss 4.2557 after 1750 stepsm failed to improve - run length 4000)
- `sequential` — Unlike 'standard' trains only one layer at any gien epoch  (best val loss 3.39 after 1000 steps, failed to improve - run length 4000)
- `elitist` — Unlike 'standard' A/B are not randomly regenerated every time, inherits 10% best from prior generation  + 90% children with noise added to 10% parents.   
- `sequential_elitist` - sequential + elitist , 3.4290 after 1000 steps, failed to improve up to 2250 steps
- `greedy`, `greedy_local` — gradient-free variants


For context. original GD achieve 1.47 after 5000 steps

## Requirements

```
pip install torch numpy
```

## Credits

Based on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the simplest, fastest repository for training medium-sized GPTs.
