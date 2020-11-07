# PyTorch's Trainer like Chainer's Trainer

We can use Trainer, Evaluator, Extension, and Reporter on PyTorch.

## Install
```bash
pip install git+https://github.com/Hiroshiba/pytorch-trainer
```

## Example
Please see [train_mnist.py](examples/train_mnist.py) that is modifyed from [Chainer's train_mnisy.py](https://github.com/chainer/chainer/blob/a45b262a4a9390044818a1d3f8cf029257ebc004/examples/mnist/train_mnist.py).
```bash
# Train with Trainer
PYTHONPATH='.' python examples/train_mnist.py \
  --device cuda \
  --autoload \
  --epoch 5
```

The logs from LogReport extension:
```
epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
0                       2.30575                              0.0768                    1.31841
1           0.213044    0.104855              0.935383       0.9668                    14.1628
2           0.0811537   0.0846022             0.97475        0.9737                    27.0918
3           0.0535006   0.0839404             0.982833       0.9749                    39.4642
4           0.0395484   0.0851855             0.987083       0.9763                    51.5603
5           0.0285093   0.0926847             0.990883       0.9726                    63.1763
```

Trainer can be saved everything, ex. Model, Optimizer, Iterator, Reporter, etc.
So using `trainer.load_state_dict`, we can resume training!
```bash
# Resume with Trainer
PYTHONPATH='.' python examples/train_mnist.py \
  --device cuda \
  --autoload \
  --epoch 10  # start from 5 epoch to 10 epoch
```
```
epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
0                       2.30575                              0.0768                    1.31841
1           0.213044    0.104855              0.935383       0.9668                    14.1628
2           0.0811537   0.0846022             0.97475        0.9737                    27.0918
3           0.0535006   0.0839404             0.982833       0.9749                    39.4642
4           0.0395484   0.0851855             0.987083       0.9763                    51.5603
5           0.0285093   0.0926847             0.990883       0.9726                    63.1763      <-- saved logs 
6           0.0257853   0.0679396             0.991567       0.9824                    75.748       <-- new logs
7           0.0202872   0.0777352             0.993583       0.9797                    85.5424
8           0.0214834   0.0869443             0.9928         0.9778                    94.8266
9           0.0166034   0.0999511             0.99435        0.9782                    104.533
10          0.0145655   0.0913273             0.995433       0.9801                    114.099
```

## Difference from Chainer.Trainer
* some extensions don't exist (ex. Schedulers or DumpGraph) 
* Trainer have Modules (because PyTorch's Optimizer don't contain Network)
* `serialize` is replaced to `state_dict` and `load_state_dict`

## Difference from Vanilla PyTorch
* cannot use DataLoader (because Trainer needs Iterator, and Iterator needs the length of dataset)

## Supported Classes
```
pytorch_trainer/
|-- iterators
|   |-- multiprocess_iterator.py
|   |-- multithread_iterator.py
|   |-- order_samplers.py
|   `-- serial_iterator.py
|-- reporter.py
`-- training
    |-- extensions
    |   |-- evaluator.py
    |   |-- exponential_shift.py
    |   |-- fail_on_nonnumber.py
    |   |-- inverse_shift.py
    |   |-- linear_shift.py
    |   |-- log_report.py
    |   |-- micro_average.py
    |   |-- multistep_shift.py
    |   |-- plot_report.py
    |   |-- polynomial_shift.py
    |   |-- print_report.py
    |   |-- progress_bar.py
    |   |-- snapshot_writers.py
    |   |-- step_shift.py
    |   |-- value_observation.py
    |   `-- warmup_shift.py
    |-- trainer.py
    |-- triggers
    |   |-- early_stopping_trigger.py
    |   |-- interval_trigger.py
    |   |-- manual_schedule_trigger.py
    |   |-- minmax_value_trigger.py
    |   |-- once_trigger.py
    |   `-- time_trigger.py
    `-- updaters
        `-- standard_updater.py
```

## Test
```bash
pytest -s -v tests
```

## TODO

- [x] Scheduler
- [ ] DataLoader

## License

[MIT LICENSE](./LICENSE) (like Chainer)
