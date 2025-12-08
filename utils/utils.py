import torch
import os
from typing import Union
from transformers import HfArgumentParser

from .configs import Arguments

from evaluations.image import ImageEvaluator
# from evaluations.molecule import MoleculeEvaluator
# from evaluations.audio import AudioEvaluator

# from diffusion.ddim import ImageSampler, MoleculeSampler
# from diffusion.audio_diffusion import AudioDiffusionSampler
# from diffusion.stable_diffusion import StableDiffusionSampler
from diffusion.stable_diffusion_video import StableDiffusionVideoSampler
from diffusion.stable_diffusion_video_normal import StableDiffusionVideoNormalSampler
from diffusion.stable_diffusion_video_standard import StableDiffusionVideoStdSampler
from diffusion.stable_diffusion_video_cogvid import StableDiffusionVideoCogVidSampler
from diffusion.stable_diffusion_video_cogvid_f16 import StableDiffusionVideoCogVidSampler as StableDiffusionVideoCogVidSampler16f
from diffusion.stable_diffusion_video_normal_doodl import StableDiffusionVideoDoodlSampler

from methods.mpgd import MPGDGuidance
from methods.lgd import LGDGuidance
from methods.base import BaseGuidance
from methods.ugd import UGDGuidance
from methods.freedom import FreedomGuidance
from methods.titanguide import TitanGuide
from methods.dps import DPSGuidance
from methods.tfg import TFGGuidance
from methods.cg import ClassifierGuidance
from methods.freedom_ori_video import FreedomGuidanceOriginalVideo
from methods.mpgd_video import MPGDGuidanceVideo
from methods.tfg_video import TFGGuidanceVideo
from methods.lgd_video import LGDGuidanceVideo
from methods.dps_video import DPSGuidanceVideo
from methods.cg_video import ClassifierGuidanceVideo
from methods.titanguide_styleguide import TitanGuideGradEstimate
from methods.doodl_algo import DOODLGuidance


import pickle


def get_logging_dir(arg_dict: dict):
    if arg_dict['guidance_name'] == 'tfg':
        # record rho, mu, sigma with scheduler
        suffix = f"rho={arg_dict['rho']}-{arg_dict['rho_schedule']}+mu={arg_dict['mu']}-{arg_dict['mu_schedule']}+sigma={arg_dict['sigma']}-{arg_dict['sigma_schedule']}"
    else:
        suffix = "guidance_strength=" + str(arg_dict['guidance_strength'])
    
    return os.path.join(
        arg_dict['logging_dir'],
        f"guidance_name={arg_dict['guidance_name']}+recur_steps={arg_dict['recur_steps']}+iter_steps={arg_dict['iter_steps']}",
        "model=" + arg_dict['model_name_or_path'].replace("/", '_'),
        "guide_net=" + arg_dict['guide_network'].replace('/', '_'),
        "target=" + str(arg_dict['target']).replace(" ", "_"),
        suffix,
    )

def get_config(add_logger=True) -> Arguments:
    args = HfArgumentParser([Arguments]).parse_args_into_dataclasses()[0]
    args.device = torch.device(args.device)

    print(">> argsguidance_strength: ", args.guidance_strength)

    if add_logger:
        from logger import setup_logger
        args.logging_dir = get_logging_dir(vars(args))
        print("logging to", args.logging_dir)
        setup_logger(args)
    
    if args.data_type == 'molecule':
        # load args
        def _get_args_gen(args_path, argse_path):
            with open(args_path, 'rb') as f:
                args_gen = pickle.load(f)
            assert args_gen.dataset == 'qm9_second_half'

            with open(argse_path, 'rb') as f:
                args_en = pickle.load(f)

            # Add missing args!
            if not hasattr(args_gen, 'normalization_factor'):
                args_gen.normalization_factor = 1
            if not hasattr(args_gen, 'aggregation_method'):
                args_gen.aggregation_method = 'sum'

            return args_gen, args_en

        args.args_gen, args.args_en = _get_args_gen(args.args_generators_path, args.args_energy_path)
    
    # examine combined guidance

    args.tasks = args.task.split('+')
    args.guide_networks = args.guide_network.split('+')
    args.targets = args.target.split('+')

    print("arg test: ", args.targets)

    assert len(args.tasks) == len(args.guide_networks) == len(args.targets)

    return args


def get_evaluator(args):

    if args.data_type == 'image':
        return ImageEvaluator(args)
    # elif args.data_type == 'molecule':
    #     return MoleculeEvaluator(args)
    elif args.data_type == 'text2image':
        return ImageEvaluator(args)
    # elif args.data_type == 'audio':
    #     return AudioEvaluator(args)
    else:
        raise NotImplementedError

def get_guidance(args, network):
    noise_fn = getattr(network, 'noise_fn', None)
    if args.guidance_name == 'no':
        return BaseGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'mpgd':
        return MPGDGuidance(args, noise_fn=noise_fn)
    elif 'ugd' in args.guidance_name:
        return UGDGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'doodl':
        return DOODLGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'freedom':
        return FreedomGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'titanguide':
        return TitanGuide(args, noise_fn=noise_fn)
    elif args.guidance_name == "freedomori":
        return FreedomGuidanceOriginalVideo(args, noise_fn=noise_fn)
    elif args.guidance_name == "titanguideest":
        return TitanGuideGradEstimate(args, noise_fn=noise_fn)
    elif args.guidance_name == 'mpgd_video':
        return MPGDGuidanceVideo(args, noise_fn=noise_fn)
    elif args.guidance_name == 'cg_video':
        return ClassifierGuidanceVideo(args, noise_fn=noise_fn)
    elif args.guidance_name == 'tfg_video':
        return TFGGuidanceVideo(args, noise_fn=noise_fn)
    elif args.guidance_name == 'dps_video':
        return DPSGuidanceVideo(args, noise_fn=noise_fn)
    elif args.guidance_name == 'lgd_video':
        return LGDGuidanceVideo(args, noise_fn=noise_fn)
    elif args.guidance_name == 'dps':
        return DPSGuidance(args, noise_fn=noise_fn)
    elif 'lgd' in args.guidance_name:
        return LGDGuidance(args, noise_fn=noise_fn)
    elif "tfg" in args.guidance_name:
        return TFGGuidance(args, noise_fn=noise_fn)
    elif 'cg' in args.guidance_name:
        return ClassifierGuidance(args, noise_fn=noise_fn)
    else:
        raise NotImplementedError

def get_network(args):
    
   
    if args.data_type == 'text2video':
        return StableDiffusionVideoSampler(args)
    elif args.data_type == "text2videostd":
        return StableDiffusionVideoStdSampler(args)
    elif args.data_type == "text2videodoodl":
        return StableDiffusionVideoDoodlSampler(args)
    elif args.data_type== 'text2videocogvid':
        return StableDiffusionVideoCogVidSampler(args)
    elif args.data_type== 'text2videocogvid16f':
        return StableDiffusionVideoCogVidSampler16f(args)
    elif args.data_type == 'text2videonormal':
        return StableDiffusionVideoNormalSampler(args)    
    else:
        raise NotImplementedError

