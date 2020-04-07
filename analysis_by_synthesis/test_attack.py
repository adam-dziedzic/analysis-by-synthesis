import torch
from torchvision.utils import make_grid

from .loss_functions import abs_loss_function
from .utils import count_correct
from foolbox_3_0_0 import foolbox
from analysis_by_synthesis.inference_attack import ABSLogits


def test_attack(model, args, device, test_loader, step, writer=None, max_batches=None, attack_name='pgd',
                attack_steps=100):
    model.eval()
    suffix = '-' + model.name if hasattr(model, 'name') else ''

    N = len(test_loader.dataset)

    loss = 0
    correct = 0

    abs_logits = ABSLogits(abs_model=model)
    abs_logits.eval()
    fmodel = foolbox.PyTorchModel(abs_logits, bounds=(0, 1), preprocessing=None)

    if attack_name == 'pgd':
        attack = foolbox.attacks.LinfPGD(steps=attack_steps)

        def fun(x, targets):
            epsilons = [0.3]
            advs, _, success = attack(fmodel, inputs=x, criterion=targets, epsilons=epsilons)
            return advs[0]

        attack_fun = fun
    else:
        raise Exception(f'Unknown attack: {attack_name}')

    for i, (data, targets) in enumerate(test_loader):
        print('batch index: ', i)
        data = data.to(device)
        targets = targets.to(device)

        data = attack_fun(x=data, targets=targets)

        logits, recs, mus, logvars = model(data)
        loss += abs_loss_function(data, targets, recs, mus, logvars, args.beta).item() * len(data)
        correct += count_correct(logits, targets)

        if i == 0 and writer is not None:
            # up to 8 samples
            n = min(data.size(0), 8)
            # flatten VAE and batch dim into a single dim
            shape = (-1,) + recs.size()[2:]
            grid = torch.cat([data[:n], recs[:, :n].reshape(shape)])
            grid = make_grid(grid, nrow=n)
            writer.add_image(f'reconstructions/test{suffix}', grid, step)

        if i == max_batches:
            # limit testing to a subset by passing max_batches
            N = i * args.test_batch_size + len(data)
            break

    loss /= N
    accuracy = 100 * correct / N
    # print(f'====> Test set: Average loss: {loss:.4f}, Accuracy: {correct}/{N} ({accuracy:.0f}%) {suffix[1:]}\n')
    data = ['test' + str(suffix), accuracy, loss]
    data_str = [str(x) for x in data]
    print(args.delimiter.join(data_str))

    if writer is not None:
        writer.add_scalar(f'loss/test{suffix}', loss, step)
        writer.add_scalar(f'accuracy/test{suffix}', accuracy, step)
        data = ['suffix', suffix, 'average loss', loss, 'accuracy', accuracy]
        data_str = [str(x) for x in data]
        writer.add_text('test summary', args.delimiter.join(data_str))
