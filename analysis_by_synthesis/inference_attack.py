import numpy as np
import torch
from torch import nn, optim

from .loss_functions import samplewise_loss_function


class AttackInference(nn.Module):
    """Takes a trained ABS model and robust inference and replaces its variational inference
    with robust inference."""

    def __init__(self, abs_model, robust_inference, n_samples=None, n_iterations=None):
        super().__init__()

        self.abs = abs_model
        self.vaes = abs_model.vaes
        self.robust_inference = robust_inference
        self.name = f'attack_{n_samples}_{n_iterations}'
        self.beta = abs_model.beta

    def attack(self, x):
        outputs = [vae(x) for vae in self.vaes]
        recs, mus, logvars = zip(*outputs)
        recs, mus, logvars = torch.stack(recs), torch.stack(mus), torch.stack(logvars)
        losses = [samplewise_loss_function(x, *output, self.beta) for output in outputs]
        losses = torch.stack(losses)
        assert losses.dim() == 2
        logits = -losses.transpose(0, 1)
        print("logits: ", logits)
        print('logits size: ', logits.size())
        # return logits, recs, mus, logvars
        return x
        #
        #
        # losses = []
        # recs = []
        # mus = []
        # for vae in self.vaes:
        #     return adv

    def forward(self, x):
        """This performs attack on the robust inference. We find the adversarial examples
        by decreasing the confidence for the VAE for the correct class and increasing the
        confidence of the 2nd best class (which is for an incorrect class). We leverage
        directly the encoder network."""

        with torch.no_grad():
            adv = self.attack(x)
            return self.robust_inference(adv)

