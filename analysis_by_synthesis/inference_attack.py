import torch
from torch import nn
from foolbox_3_0_0 import foolbox


class ABSLogits(nn.Module):
    """ABS model wrapper that only returns the logits."""

    def __init__(self, abs_model):
        super().__init__()
        self.abs_model = abs_model

    def forward(self, x):
        logits, recs, mus, logvars = self.abs_model.forward(x)
        return logits


class AttackInference(nn.Module):
    """Takes a trained ABS model and robust inference and replaces its variational inference
    with robust inference."""

    def __init__(self, abs_model, robust_inference, n_samples=None, n_iterations=None, attack_name='pgd',
                 attack_steps=100):
        super().__init__()
        self.abs = abs_model
        self.abs_logits = ABSLogits(abs_model=abs_model)
        self.abs_logits.eval()
        self.fmodel = foolbox.models.pytorch.PyTorchModel(self.abs_logits, bounds=(0, 1), preprocessing=None)
        self.vaes = abs_model.vaes
        self.robust_inference = robust_inference
        self.name = f'attack_{n_samples}_{n_iterations}'
        self.beta = abs_model.beta
        self.attack_name = attack_name
        self.attack_steps = attack_steps

    def test_clean(self):
        # get data and test the model
        batchsize = 20
        dataset = 'mnist'
        images, labels = foolbox.samples(self.fmodel, dataset=dataset, batchsize=batchsize)
        print('clean accuracy: ', foolbox.accuracy(self.fmodel, images, labels))

    def attack(self, x, labels=None):
        if labels is None:
            labels = self.abs_model(x)
        # apply the attack
        if self.attack_name == 'pgd':
            attack = foolbox.attacks.LinfPGD(steps=self.attack_steps)
            # epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
            epsilons = [0.3]
            advs, _, success = attack(self.fmodel, inputs=self.images, criterion=labels, epsilons=epsilons)
            advs = advs[0]
            return advs

    def forward(self, x):
        """This performs an attack on the robust inference. We find the adversarial examples
        by decreasing the confidence for the VAE for the correct class and increasing the
        confidence of the 2nd best class (which is for an incorrect class). We leverage
        directly the encoder network."""

        with torch.no_grad():
            adv = self.attack(x)
            return self.robust_inference(adv)


if __name__ == "__main__":
    import os

    print('current working directory: ', os.getcwd())
    # create the ABS model
    from analysis_by_synthesis.architecture import ABS
    from analysis_by_synthesis.args import get_args

    args = get_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ABS(n_classes=10, n_latents_per_class=8, beta=args.beta).to(device)
    model.eval()
    model.load_state_dict(torch.load("../" + args.load))

    from analysis_by_synthesis.inference_robust import RobustInference

    # create wrappers that perform robust inference
    kwargs = {
        'fraction_to_dismiss': args.fraction_to_dismiss,
        'lr': args.inference_lr,
        'radius': args.clip_to_sphere,
    }
    n_samples = 80
    n_iterations = 0
    robust_inference1 = RobustInference(model, device, n_samples=n_samples, n_iterations=n_iterations, **kwargs)
    attack_inference1 = AttackInference(abs_model=model, robust_inference=robust_inference1, n_samples=n_samples,
                                        n_iterations=n_iterations)

    attack_inference1.test_clean()
