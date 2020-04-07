import logging
import time
import unittest

import torch

from analysis_by_synthesis.architecture import ABS
from analysis_by_synthesis.args import get_args
from analysis_by_synthesis.args import print_args
from analysis_by_synthesis.datasets import get_dataset, get_dataset_loaders
from analysis_by_synthesis.inference_attack import ABSLogits
from analysis_by_synthesis.inference_robust import RobustInference
from analysis_by_synthesis.utils import count_correct
from analysis_by_synthesis.utils.logging_utils import get_logger
from analysis_by_synthesis.utils.logging_utils import set_up_logging
from foolbox_3_0_0 import foolbox

ERR_MSG = "Expected x is different from computed y."


class TestInferenceAttack(unittest.TestCase):

    def setUp(self):
        log_file = "inference_attack_test.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")
        seed = 31
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("cuda is available")
            torch.cuda.manual_seed_all(seed)
        else:
            self.device = torch.device("cpu")
            print("cuda is not available")
            torch.manual_seed(seed)
        self.dtype = torch.float
        self.ERR_MESSAGE_ALL_CLOSE = "The expected array desired and " \
                                     "computed actual are not almost equal."
        self.args = get_args()
        self.init_abs()

        self.batchsize = 20
        self.dataset = 'mnist'
        self.images, self.labels = foolbox.samples(self.fmodel, dataset=self.dataset, batchsize=self.batchsize)

        # create wrappers that perform robust inference
        kwargs = {
            'fraction_to_dismiss': self.args.fraction_to_dismiss,
            'lr': self.args.inference_lr,
            'radius': self.args.clip_to_sphere,
        }
        n_samples = 80
        n_iterations = 0
        self.robust_inference1 = RobustInference(self.abs, self.device, n_samples=n_samples,
                                                 n_iterations=n_iterations, **kwargs)
        self.robust_inference2 = RobustInference(self.abs, self.device, n_samples=8000,
                                                 n_iterations=0, **kwargs)
        self.robust_inference3 = RobustInference(self.abs, self.device, n_samples=8000,
                                                 n_iterations=50, **kwargs)

    def init_abs(self):
        self.abs = ABS(n_classes=10, n_latents_per_class=8, beta=self.args.beta).to(self.device)
        self.abs.eval()
        self.abs.load_state_dict(torch.load("../" + self.args.load))
        self.abs_logits = ABSLogits(abs_model=self.abs)
        self.abs_logits.eval()
        self.fmodel = foolbox.PyTorchModel(self.abs_logits, bounds=(0, 1), preprocessing=None)

    def test_forward_clean_data(self):
        print('clean accuracy: ', foolbox.accuracy(self.fmodel, self.images, self.labels))

    def test_pgd_attack(self):
        print('pgd attack')
        attack = foolbox.attacks.LinfPGD()
        # epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
        epsilons = [0.3]
        advs, _, success = attack(self.fmodel, inputs=self.images, criterion=self.labels, epsilons=epsilons)

        # calculate and report the robust accuracy
        print('robust accuracy')
        print('epsilon, accuracy')
        robust_accuracy = 1 - success.to(torch.float32).mean(axis=-1)
        for eps, acc in zip(epsilons, robust_accuracy):
            print(eps, ', ', acc.item())

        for epsilon, adv_data in zip(epsilons, advs):
            print(f'epsilon: {epsilon}, pgd accuracy: {foolbox.accuracy(self.fmodel, adv_data, self.labels)}')

    def get_robust_accuracy(self, robust_inference, data, labels):
        logits, recs, mus, logvars = robust_inference(data)
        robust_correct = count_correct(predictions=logits, labels=labels)
        robust_accuracy = robust_correct / len(data)
        return robust_accuracy

    def test_robust_inference_on_clean_data(self):
        print('robust accuracy on clean data')
        for nr, robust_inference in [(1, self.robust_inference1),
                                     (2, self.robust_inference2),
                                     (3, self.robust_inference3)]:
            robust_accuracy = self.get_robust_accuracy(robust_inference=robust_inference,
                                                       data=self.images, labels=self.labels)
            print(f'robust_inference{nr} accuracy: {robust_accuracy}')

    def test_robust_inference_against_pgd(self):
        print('pgd attack')
        attack = foolbox.attacks.LinfPGD(steps=100)
        # epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
        epsilons = [0.3]
        advs, _, success = attack(self.fmodel, inputs=self.images, criterion=self.labels, epsilons=epsilons)

        for epsilon, adv_data in zip(epsilons, advs):
            pgd_accuracy = foolbox.accuracy(self.fmodel, adv_data, self.labels)
            print(f'epsilon: {epsilon}, pgd accuracy: {pgd_accuracy}')

            for nr, robust_inference in [(1, self.robust_inference1),
                                         (2, self.robust_inference2),
                                         (3, self.robust_inference3)]:
                robust_accuracy = self.get_robust_accuracy(robust_inference=robust_inference,
                                                           data=adv_data, labels=self.labels)
                print(f'robust_inference{nr} accuracy: {robust_accuracy}')

    def attack_mnist(self, robust_inference=None, nr=None, epsilon=0.3):
        # load the train and test set
        args = self.args
        print_args(args=args)
        use_cuda = True
        train_set, test_set = get_dataset(args.dataset, args.no_augmentation)
        train_loader, test_loader = get_dataset_loaders(train_set, test_set, use_cuda, args)

        if robust_inference is None:
            robust_inference = self.robust_inference1
            nr = 1
        print(f'robust_inference{nr} in attack_mnist (all data samples), PGD attack steps {args.attack_steps}:')

        attack = foolbox.attacks.LinfPGD(steps=args.attack_steps)

        epsilons = [epsilon]

        correct = 0
        correct_adv = 0
        counter = 0

        for i, (data, targets) in enumerate(test_loader):
            print('batch number: ', i)
            data = data.to(self.device)
            targets = targets.to(self.device)

            counter += len(targets)

            logits, recs, mus, logvars = robust_inference(data)
            correct += count_correct(predictions=logits, labels=targets)

            advs, _, _ = attack(self.fmodel, inputs=data, criterion=targets, epsilons=epsilons)
            advs = advs[0] # we only have 1 epsilon value

            logits, recs, mus, logvars = robust_inference(advs)
            correct_adv += count_correct(predictions=logits, labels=targets)

        accuracy = correct / counter
        print('accuracy mnist on clean data: ', accuracy)

        accuracy_adv = correct_adv / counter
        print('accuracy mnist on adv data: ', accuracy_adv)

    def test_attack_mnist_many_robust_inference_tests(self):
        epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.34, 0.4, 0.5, 1.0]
        for epsilon in epsilons:
            for nr, robust_inference in [
                # (1, self.robust_inference1),
                # (2, self.robust_inference2),
                (3, self.robust_inference3),
            ]:
                start = time.time()
                self.attack_mnist(robust_inference=robust_inference, nr=nr, epsilon=epsilon)
                stop = time.time()
                print('Elapsed time (sec): ', stop - start)


if __name__ == '__main__':
    unittest.main()
