from tqdm import tqdm
from einops import rearrange
import numpy as np
import torch
from torchvision import transforms
import glob
import torch
import torchvision
from torchvision.io import VideoReader
torchvision.set_video_backend("video_reader")
import os
# from ..dists import frechet_distance
import itertools
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
from style_CLIP import StyleCLIP
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE



@torch.no_grad()
def _compute_style_score(images, targets, clip_embedding):
	
	similarity_list = []

	for target in targets:

		target_embed = clip_embedding.get_target_embedding(target).cuda()
		image_embed = []
		
		for bs in range(0, len(images) ):
			# imgs = [clip_embedding.to_tensor(img) for img in images[bs:bs+len(images)]]
			# print(">> >images: ", images[0].shape) ##  torch.Size([3, 256, 256])
			imgs = [img.unsqueeze(0) for img in images[bs:bs+20]]
			image_embed.append(clip_embedding.get_gram_matrix(torch.concat(imgs, dim=0).cuda()))
		
		image_embed = torch.concat(image_embed, dim=0)
		diff = (image_embed - target_embed).reshape(image_embed.size(0), -1)
		similarity_list.append(-(diff ** 2).sum(dim=1).sqrt() / 10)

	return torch.cat(similarity_list, dim=0).sum().item()#torch.cat(similarity_list, dim=0).mean().item()


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
 
	video_frames = torch.stack(video_frames)  # (b, t, c, h, w)
	
	return video_frames, audio_frames

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


def preprocess_videos(videos, sequence_length=None):
	# video: BCTHW, [0, 1]
	assert videos.ndim == 5
	b, c, t, h, w = videos.shape
	# temporal crop
	if sequence_length is not None:
		assert sequence_length <= t
		videos = videos[:, :, :sequence_length]
		videos = videos.contiguous()
		t = sequence_length
	
	videos = rearrange(videos, "b c t h w -> (b t) c h w")
	
	transform_func = transforms.Compose([
		transforms.Resize(
			(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
		),
		transforms.CenterCrop(224)
	])
	videos = transform_func(videos)
	videos = videos * 2 - 1
	
	videos = rearrange(videos, "(b t) c h w -> b c t h w", t=t)
	
	return videos

@torch.no_grad()
def compute_fvd_video_features(videos, net):
	'''
		videos in shape BCTHW in [0., 1.]
	'''
	videos = preprocess_videos(videos).contiguous()
	
	detector_kwargs = dict(rescale=False, resize=False, return_features=True)
	logits = net(videos, **detector_kwargs)
	# logits = net(videos)
	return logits

def compute_style_directly(video_paths ,target='/mnt/data2/chris/code/Training-Free-Guidance/data/wikiart/0.png'):
	fvd_features_arr = []
	vid_tensors = []
	limit_size = 100
	sum_score = 0
	count_score=  0
	clip_embedding = StyleCLIP(device="cuda", network='openai/clip-vit-base-patch32')

	image_embeds = []

	print(">> video_paths: ", video_paths)

	for video_path in video_paths:
		video_tensors, audio_tensors = load_av_clips_uniformly(
			video_path=video_path,
				video_fps=video_fps,
				video_num_frame=video_num_frame,
				image_size=image_size,
				num_clips=1,
				load_audio_as_melspectrogram=True)
		# print(">>vid_tensors: ", video_tensors[0].shape, len(video_tensors)) #vid_tensors[0].shape)
		# if len(vid_tensors) >= 10:
		img_tensors = video_tensors[0] #torch.cat(vid_tensors[0], dim=0)
		# print(">>img_tensors: ", img_tensors.shape) # torch.Size([176, 3, 256, 256])
		# sum_score += _compute_style_score(img_tensors, targets=[target], clip_embedding=clip_embedding)
		# count_score += len(img_tensors)
		# for target in img_tensors:
		# print(">>img_tensors: ", img_tensors.shape)
		# 	# target_embed = clip_embedding.get_target_embedding(target).cuda()
		# 	image_embed = []
			
		# for bs in range(0, len(images) ):
		# 	# imgs = [clip_embedding.to_tensor(img) for img in images[bs:bs+len(images)]]
			# print(">> >images: ", images[0].shape) ##  torch.Size([3, 256, 256])
		imgs = [img.unsqueeze(0) for img in img_tensors]
		embed = clip_embedding.get_embedding_feature(img_tensors).cuda()
		# print(">> embed: ", embed.shape) ##  torch.Size([16, 49, 768])
		image_embeds.append(embed.mean(1).mean(0).squeeze().unsqueeze(0))
			
		# image_embed = torch.concat(image_embed, dim=0)

		# vid_tensors = []
	
		# vid_tensors.append(video_tensors[0])
	
	#### add target:
	print("tARGET : ", target)
	target_embed = clip_embedding.get_target_embedding_feature(target).cuda()
	print(">>target_embed: ", target_embed.shape)
	image_embeds.append(target_embed.mean(1).mean(0).squeeze().unsqueeze(0))
	
	image_embeds = torch.cat(image_embeds, dim=0)
	print(">>> image_embeds: ", image_embeds.shape)
	X = image_embeds.cpu().numpy()
	n_components=2
	perplexity=5
	tsne = TSNE(n_components=n_components, init='random', random_state=0, perplexity=perplexity)
	# y_subset = np.ones([10])
	X_tsne = tsne.fit_transform(X)

	plt.figure(figsize=(10, 10))

	# 各クラス（0〜9）のプロット
	# for i in range(10):
	# 	# 各ラベルのデータを取得
	# 	indices = y_subset == i
		# plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=str(i), alpha=0.7)
	plt.scatter(X_tsne[:-1, 0], X_tsne[:-1, 1], label="test", alpha=0.7)
	plt.scatter(X_tsne[-1, 0], X_tsne[-1, 1], label="reference", alpha=0.7)

	# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], label="test", alpha=0.7)
	
	plt.legend()
	plt.title("t-SNE Visualization")
	plt.xlabel("t-SNE Dimension 1")
	plt.ylabel("t-SNE Dimension 2")
	plt.grid(True)
	plt.savefig("../logs/tsne.png")
	# print("SCORE: ", sum_score/count_score, sum_score, count_score)

# def compute_style_directly(video_paths ,target='/mnt/data2/chris/code/Training-Free-Guidance/data/wikiart/0.png'):
# 	fvd_features_arr = []
# 	vid_tensors = []
# 	limit_size = 100
# 	sum_score = 0
# 	count_score=  0
# 	clip_embedding = StyleCLIP(device="cuda", network='openai/clip-vit-base-patch32')

# 	for video_path in video_paths:
# 		video_tensors, audio_tensors = load_av_clips_uniformly(
# 			video_path=video_path,
# 				video_fps=video_fps,
# 				video_num_frame=video_num_frame,
# 				image_size=image_size,
# 				num_clips=1,
# 				load_audio_as_melspectrogram=True)
# 		# print(">>vid_tensors: ", video_tensors[0].shape) #vid_tensors[0].shape)
# 		if len(vid_tensors) >= 10:
# 			img_tensors = torch.cat(vid_tensors, dim=0)
# 			# print(">>img_tensors: ", img_tensors.shape) # torch.Size([176, 3, 256, 256])
# 			sum_score += _compute_style_score(img_tensors, targets=[target], clip_embedding=clip_embedding)
# 			count_score += len(img_tensors)
# 			vid_tensors = []
		
# 		vid_tensors.append(video_tensors[0])

# 	print("SCORE: ", sum_score/count_score, sum_score, count_score)
#  vid_tensors


if __name__== "__main__":
	device = "cuda:0"
	dtype = torch.float16
	# i3d = load_i3d_pretrained(device).to(device=device, dtype=dtype)

	gt_video_root="/mnt/data2/chris/datasets/vggsound/reference_videos_2secs/"
	generated_video_root=  "/mnt/data2/chris/outputs/video_gen_guide/style_transfer/titan_gradest_cogvid_tsne_ablation/" #"/mnt/data2/chris/outputs/vgg/test_ours_epsdiff/" ##"/mnt/data2/chris/outputs/vgg/test_seeinghearing/"
	gt_video_paths = glob.glob(f"{gt_video_root}/*.mp4")
	generated_video_paths = glob.glob(f"{generated_video_root}/*.mp4")
	image_size = (256, 256)
	video_fps= 8
	video_num_frame = 16
	 ### ours: -916.0100517578124 -14656160.828125 16000, 
	compute_style_directly(generated_video_paths)
	### titan: -916.0100517578124 -14656160.828125 16000
    # groundtruth_fvd_features = read_videos(gt_video_paths)
    # generated_fvd_features = read_videos(generated_video_paths)
    
    # groundtruth_fvd_features = torch.cat(groundtruth_fvd_features)
    # generated_fvd_features = torch.cat(generated_fvd_features)
    # # kvd_score = polynomial_mmd(groundtruth_fvd_features, generated_fvd_features)# fvd_score = frechet_distance(groundtruth_fvd_features, generated_fvd_features)
	

    # print("KVD: ", kvd_score)
	## snh KVD:  28.925474559468512
	## ours eps: KVD:  29.36223087292774