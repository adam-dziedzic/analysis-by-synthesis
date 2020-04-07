import logging
import unittest

import torch

from analysis_by_synthesis.architecture import ABS
from .utils.logging_utils import get_logger
from .utils.logging_utils import set_up_logging
import foolbox

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

    def test_forward(self):
        # create the ABS model
        num_classes = 10
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = ABS(n_classes=num_classes, n_latents_per_class=8, beta=1).to(device)
        model.num_classes = num_classes
        model.eval()
        bounds = foolbox.utils.Bounds(0, 1)
        fmodel = foolbox.models.pytorch.PyTorchModel(model=model, bounds=(0, 1), preprocessing=None, device=device)

        batchsize = 20
        dataset = 'mnist'
        images, labels = foolbox.samples(fmodel, dataset=dataset, batchsize=batchsize)
        clean_accuracy = foolbox.accuracy(fmodel=fmodel, inputs=images, labels=labels)
        print('clean_accuracy: ', clean_accuracy)

