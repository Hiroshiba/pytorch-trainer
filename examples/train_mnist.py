#!/usr/bin/env python
import argparse

import torch
from torchvision.transforms import transforms

import chainer
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets
from chainer import training
from chainer.training import extensions
from chainer import reporter

import matplotlib
matplotlib.use('Agg')


# Network definition
class MLP(nn.Module):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(n_in, n_units)  # n_in -> n_units
        self.l2 = nn.Linear(n_units, n_units)  # n_units -> n_units
        self.l3 = nn.Linear(n_units, n_out)  # n_units -> n_out

    def forward(self, x):
        x = x.view((len(x), -1))
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return F.log_softmax(self.l3(h2), dim=1)


def accuracy(y, t):
    pred = y.argmax(axis=1).reshape(t.shape)
    acc = (pred == t).mean(dtype=y.dtype)
    return acc


class Classifier(nn.Module):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        self.predictor = predictor

    def forward(self, x, t):
        y = self.predictor(x)
        loss = F.nll_loss(y, t)
        reporter.report({'loss': loss}, self)
        acc = accuracy(y, t)
        reporter.report({'accuracy': acc}, self)
        return loss


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--autoload', action='store_true',
                        help='Automatically load trainer snapshots in case'
                        ' of preemption or other temporary system failure')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = torch.device(args.device)

    print('Device: {}'.format(device))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = Classifier(MLP(784, args.unit, 10))
    model.to(device)

    # Setup an optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train = datasets.MNIST('data', train=True, download=True, transform=transform)
    test = datasets.MNIST('data', train=False, transform=transform)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, model, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=device),
                   call_before_training=True)

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    # trainer.extend(extensions.DumpGraph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    # Take a snapshot each ``frequency`` epoch, delete old stale
    # snapshots and automatically load from snapshot files if any
    # files are already resident at result directory.
    trainer.extend(extensions.snapshot(n_retains=1, autoload=args.autoload),
                   trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(), call_before_training=True)

    # Save two plot images to the result dir
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'),
        call_before_training=True)
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'),
        call_before_training=True)

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']),
        call_before_training=True)

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume is not None:
        # Resume from a snapshot (Note: this loaded model is to be
        # overwritten by --autoload option, autoloading snapshots, if
        # any snapshots exist in output directory)
        trainer.load_state_dict(torch.load(args.resume))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
