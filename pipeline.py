import os
import numpy as np
import PIL.Image as Image
from abc import ABC, abstractmethod
from diffusion.base import BaseSampler
from methods.base import BaseGuidance
from evaluations.base import BaseEvaluator
from utils.configs import Arguments
import logger

class BasePipeline(object):
    def __init__(self,
                 args: Arguments, 
                 network: BaseSampler, 
                 guider: BaseGuidance, 
                 evaluator: BaseEvaluator):
        self.network = network
        self.guider = guider
        self.evaluator = evaluator
        self.logging_dir = args.logging_dir
        self.check_done = args.check_done

    @abstractmethod
    def sample(self, sample_size: int):
        
        # samples = self.check_done_and_load_sample()
        samples = None
        
        if samples is None:
            samples = self.network.sample(sample_size=sample_size, guidance=self.guider)
            # samples = self.network.tensor_to_obj(samples)
        
        return samples
    
    def evaluate(self, samples):
        return self.check_done_and_evaluate(samples)
    
    def check_done_and_evaluate(self, samples):
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, 'metrics.json')):
            logger.log("Metrics already generated. To regenerate, please set `check_done` to `False`.")
            return None
        return self.evaluator.evaluate(samples)

    def check_done_and_load_sample(self):
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, "finished_sampling")):
            logger.log("found tags for generated samples, should load directly. To regenerate, please set `check_done` to `False`.")
            return logger.load_samples()

        return None

