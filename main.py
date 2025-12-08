from utils.utils import get_config, get_evaluator, get_guidance, get_network
from pipeline import BasePipeline
import torch
import logger

if __name__ == '__main__':
    # Please tsee utils/config.py for the complete argument lists
    args = get_config()
    ## prepare core modules based on configs ##
    
    # Unconditional generative model
    network = get_network(args)
    # guidance method encoded by prediction model
    guider = get_guidance(args, network)
    # evaluator for generated samples
    try:
        evaluator = get_evaluator(args)
    except NotImplementedError:
        evaluator = None

    pipeline = BasePipeline(args, network, guider, evaluator)

    samples = pipeline.sample(args.num_samples)
    logger.log_samples(samples)
    
    torch.cuda.empty_cache()
 