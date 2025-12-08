import os
import argparse
import numpy as np
import torch
# import pandas as pd
import glob
from tqdm import tqdm
from PIL import Image
import torchvision
from torchvision.io import VideoReader
torchvision.set_video_backend("video_reader")

import torchvision.transforms as T
from transformers import AutoProcessor, CLIPModel, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from scipy import linalg

transform = T.ToPILImage()


def load_video_clip_from_videoreader(
		av_reader,
		clip_start_timestamp,
		clip_duration,
		video_fps,
		video_num_frame,
		image_size,
		flip=False,
		randcrop=False,
		normalize=False
):
	av_reader.set_current_stream("video")
	keyframe_coverage = 1. / video_fps
	
	video_frames = []
	frame_timestamp = clip_start_timestamp
	for i, frame in enumerate(itertools.takewhile(
			lambda x: x['pts'] <= clip_start_timestamp + clip_duration + keyframe_coverage / 2.,
			av_reader.seek(max(clip_start_timestamp, 0.))
	)):
		if frame["pts"] >= frame_timestamp:
			video_frames.append(frame["data"])  # (c, h, w) tensor [0, 255]
			frame_timestamp += keyframe_coverage
		
		if len(video_frames) == video_num_frame:
			break
	
	if len(video_frames) < video_num_frame:
		res_length = video_num_frame - len(video_frames)
		for _ in range(res_length):
			video_frames.append(video_frames[-1])
	
	video_frames = torch.stack(video_frames, dim=0).float() / 255.
	
	# video_frames = load_and_transform_images_stable_diffusion(
	# 	video_frames,
	# 	size=image_size,
	# 	flip=False,
	# 	randcrop=False,
	# 	normalize=normalize
	# ).float()  # (n_frame, 3, h, w) in range [0., 1.]
	
	return video_frames


def load_av_clips_uniformly(
		video_path: str,
		video_fps: int = 8,
		video_num_frame: int = 16,
		image_size  = 256,
		num_clips: int = 1,
		load_audio_as_melspectrogram: bool = False,
):
	'''
	Return:
		video_frames: (b f c h w) in [0, 1]
		audio_frames:
			if load_audio_as_melspectrogram is True: (b 1 n t)
			else: List of tensors (b c ti), ti can be different
	'''
	clip_duration = video_num_frame / video_fps
	av_reader = VideoReader(video_path, stream="video")
	meta_data = av_reader.get_metadata()
	video_duration, orig_video_fps = float(meta_data["video"]["duration"][0]), float(meta_data["video"]["fps"][0])
	# audio_duration, audio_sr = float(meta_data["audio"]["duration"][0]), int(meta_data["audio"]["framerate"][0])
	av_duration = video_duration #min(video_duration, audio_duration)
	# assert av_duration >= clip_duration, [video_path, video_duration, audio_duration]
	
	# 1. Sample clip start times
	if num_clips == 1:
		clip_start_timestamps = np.array([(av_duration - clip_duration) / 2.])
	else:
		clip_start_timestamps = np.linspace(0., av_duration - clip_duration, endpoint=True, num=num_clips)
	
	video_frames = []
	audio_frames = []
	for clip_start_timestamp in clip_start_timestamps:
		video_frames.append(
			load_video_clip_from_videoreader(
				av_reader,
				clip_start_timestamp,
				clip_duration,
				video_fps,
				video_num_frame,
				image_size,
				flip=False,
				randcrop=False,
				normalize=False
			)
		)
		# audio_frames.append(
		# 	load_audio_clip_from_videoreader(
		# 		av_reader,
		# 		clip_start_timestamp,
		# 		clip_duration,
		# 		audio_sr,
		# 		load_audio_as_melspectrogram
		# 	)
		# )
	
	video_frames = torch.stack(video_frames)  # (b, t, c, h, w)
	# if load_audio_as_melspectrogram:
	# 	audio_frames = torch.stack(audio_frames)  # (b, 1, c, t)
	
	return video_frames, audio_frames

# Clip score for text alignment
def clip_score_text(frames, prompt):

    inputs = processor(text=[prompt], images=frames, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image.detach().cpu().numpy()
    score = logits_per_image.mean()

    return score


# Clip score for frame consistency
def clip_score_frame(frames):

    inputs = processor(images=frames, return_tensors="pt").to(device)

    print(">> inputs: " , inputs['pixel_values'].shape)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).detach().cpu().numpy()

    cosine_sim_matrix = cosine_similarity(image_features)
    np.fill_diagonal(cosine_sim_matrix, 0)  # set diagonal elements to 0
    print(">> len(frames): ", len(frames))
    score = cosine_sim_matrix.sum() / (len(frames) * (len(frames)-1))

    return score


# PickScore: https://github.com/yuvalkirstain/PickScore
def pick_score(frames, prompt):

    image_inputs = processor(images=frames, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    text_inputs = processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        score_per_image = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        score_per_image = score_per_image.detach().cpu().numpy()
        score = score_per_image.mean()

    return score


eval_functions = {
    # "clip_score_text": clip_score_text,
    "clip_score_frame": clip_score_frame,
    "pick_score": pick_score,
}

## RUN : python compute_clip_frame.py --metrics clip_score_frame
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--submission_path", type=str, required=True, help="path to submission folder")
    parser.add_argument("--data_path", type=str, default="./data/loveu-tgve-2023", help="path to data folder")
    parser.add_argument("--metric", type=str, default="clip_score_frame",
                        choices=['clip_score_text', 'clip_score_frame', 'pick_score'])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    width, height = 480, 480
    device = args.device

    if args.metric == "clip_score_text" or args.metric == "clip_score_frame":
        preatrained_model_path = "openai/clip-vit-large-patch14"
        model = CLIPModel.from_pretrained(preatrained_model_path).to(device)
        processor = AutoProcessor.from_pretrained(preatrained_model_path)
    elif args.metric == "pick_score":
        preatrained_model_path = "pickapic-anonymous/PickScore_v1"
        model = AutoModel.from_pretrained(preatrained_model_path).to(device)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        processor = AutoProcessor.from_pretrained(processor_path)
    else:
        raise NotImplementedError(args.metric)

    # df = pd.read_csv(f"{args.data_path}/LOVEU-TGVE-2023_Dataset.csv")
    # sub_dfs = {
    #     'DAVIS_480p': df[1:17],
    #     'youtube_480p': df[19:42],
    #     'videvo_480p': df[44:82],
    # }

    scores = []
    eval_function = eval_functions[args.metric]
    # generated_video_root = "/mnt/data2/chris/outputs/video_gen_guide/imagebind/titan384x384/" #"/mnt/data2/chris/outputs/vgg/test_seeingandhearing_textupd/"
    generated_video_root = "/mnt/data2/chris/outputs/video_gen_guide/video_inpaint/doodl/"
    generated_video_paths = glob.glob(f"{generated_video_root}/*.mp4")

    width, height = 256, 256
    video_fps = 8
    video_num_frame = 16
    image_size = 256


    for video_path in generated_video_paths:
        video_tensors, audio_tensors = load_av_clips_uniformly(
                video_path=video_path,
                video_fps=video_fps,
                video_num_frame=video_num_frame,
                image_size=image_size,
                num_clips=1,
                load_audio_as_melspectrogram=False
            )


        # fvd_features = compute_fvd_video_features(
        #                 rearrange(video_tensors, "b f c h w -> b c f h w"),
        #                 i3d
        #             ).detach().cpu() # (b, c)
        # generated_fvd_features.append(fvd_features)    
        video_tensors= video_tensors.to(dtype=torch.float16, device="cuda:0")
        video_tensors = video_tensors.squeeze()
        print(">> video_tensors.shape: ", video_tensors.shape)
        frames = [transform(x[0]) for x in video_tensors]
        frames = [i.resize((width, height)) for i in frames]
        score = eval_function(frames)
        print(">> score: " , score)
        scores.append(score)

        print(" wkwkw {:.3f}".format(  np.mean(scores)))

    # for sub_name, sub_df in sub_dfs.items():
    #     print(f"Processing {sub_name} ..")
    #     for index, row in tqdm(sub_df.iterrows(), total=sub_df.shape[0]):
    #         video_name = row['Video name']
    #         edited_prompts = {x.split(" ")[0].lower(): str(row[x]).strip() for x in [
    #             "Style Change Caption",
    #             "Object Change Caption",
    #             "Background Change Caption",
    #             "Multiple Changes Caption"
    #         ]}

    #         for edited_type, edited_prompt in edited_prompts.items():
    #             video_path = f"{args.submission_path}/{sub_name}/{video_name}/{edited_type}"
    #             if not os.path.exists(video_path):
    #                 raise FileNotFoundError(video_path)
    #             frames = [Image.open(x) for x in sorted(glob(f"{video_path}/*.jpg"))]
    #             frames = [i.resize((width, height)) for i in frames]

    #             scores.append(eval_function(frames, edited_prompt))

    print("{}: {:.3f}".format(args.metric, np.mean(scores)))