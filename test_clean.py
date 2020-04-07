#!/usr/bin/env python3
import os

import torch
import torchvision
from tensorboardX import SummaryWriter

from analysis_by_synthesis.architecture import ABS
from analysis_by_synthesis.args import get_args
from analysis_by_synthesis.datasets import get_dataset, get_dataset_loaders
from analysis_by_synthesis.inference_robust import RobustInference
from analysis_by_synthesis.sample import sample
from analysis_by_synthesis.test import test


def main():
    assert not hasattr(torchvision.datasets.folder, 'find_classes'), 'torchvision master required'

    args = get_args()

    if args.test_only:
        args.initial_evaluation = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load the train and test set
    train_set, test_set = get_dataset(args.dataset, args.no_augmentation)
    train_loader, test_loader = get_dataset_loaders(train_set, test_set, use_cuda, args)

    step = 0

    # create the ABS model
    model = ABS(n_classes=10, n_latents_per_class=8, beta=args.beta).to(device)
    model.eval()

    # load weights
    print('current working directory: ', os.getcwd())
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))
    else:
        raise Exception("Please provide the 'load' argument with a valid path to a model.")

    # create wrappers that perform robust inference
    kwargs = {
        'fraction_to_dismiss': args.fraction_to_dismiss,
        'lr': args.inference_lr,
        'radius': args.clip_to_sphere,
    }
    robust_inference1 = RobustInference(model, device, n_samples=80, n_iterations=0, **kwargs)
    robust_inference2 = RobustInference(model, device, n_samples=8000, n_iterations=0, **kwargs)
    robust_inference3 = RobustInference(model, device, n_samples=8000, n_iterations=50, **kwargs)

    # create writer for TensorBoard
    writer = SummaryWriter(args.logdir) if args.logdir is not None else None

    # model changed, so make sure reconstructions are regenerated
    robust_inference1.invalidate_cache()
    robust_inference2.invalidate_cache()
    robust_inference3.invalidate_cache()

    # common params for calls to test
    params = (args, device, test_loader, step, writer)

    header = ['suffix', 'average accuracy', 'loss']
    print(args.delimiter.join(header))
    # some evaluations can happen after every epoch because they are cheap
    test(model, *params)
    # test(robust_inference1, *params)
    # test(robust_inference2, *params)
    # test(robust_inference3, *params)

    sample(model, device, step, writer)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()

