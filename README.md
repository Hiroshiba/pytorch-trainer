# PyTorch's Trainer like Chainer's Trainer

We can use Trainer, Evaluator, Extension, and Reporter on PyTorch.

## Example
Please see [train_mnist.py](examples/train_mnist.py) that is modifyed from [Chainer's train_mnisy.py](https://github.com/chainer/chainer/blob/a45b262a4a9390044818a1d3f8cf029257ebc004/examples/mnist/train_mnist.py).
```bash
PYTHONPATH='.' python examples/train_mnist.py \
  --device cuda \
  --autoload \
  --epoch 10

```

## Difference from Chainer.Trainer
* some extensions don't exist (ex. Shedulers or DumpGraph) 
* Trainer have Modules (because PyTorch's Optimizer don't contain Network)
* `serialize` is replaced to `state_dict` and `load_state_dict`

## Difference from Vanilla PyTorch
* cannot use DataLoader (because Trainer needs Iterator, and Iterator needs the length of dataset)

## Test
```bash
pytest -s -v tests
```

## TODO

- [ ] Scheduler
- [ ] DataLoader

## License

MIT LICENSE (like Chainer)
