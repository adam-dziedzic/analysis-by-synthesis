import torch
from torchvision.utils import make_grid

from .loss_functions import abs_loss_function
from .utils import count_correct


def test(model, args, device, test_loader, step, writer=None, max_batches=None):
    model.eval()
    suffix = '-' + model.name if hasattr(model, 'name') else ''

    N = len(test_loader.dataset)

    loss = 0
    correct = 0

    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)
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
