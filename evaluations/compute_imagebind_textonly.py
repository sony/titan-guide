
from imagebind.imagebind_data import load_and_transform_audio_data, load_and_transform_video_data, load_and_transform_text, load_and_transform_vision_data, waveform2melspec
from imagebind.imagebind.models import imagebind_model
from imagebind.imagebind.models.imagebind_model import ModalityType
import pandas
import csv
import glob
import torch
import torch.nn.functional as F

def get_batch_paths(chunk_generated_video_paths, ref_audio_root):
    video_paths = []
    audio_paths= []
    text_list=  []

    for video_path in chunk_generated_video_paths:
        # video_name = video_path.split("/")[-1]
        video_paths.append(video_path)
        idd = video_path.split("/")[-1].replace(".mp4","")
        text_list.append(idd) ## id is text
        # audio_path = ref_audio_root +"/" + idd + ".wav"
        # print("audio_path: ", idd, audio_path, text_ref[idd])
        # audio_paths.append(audio_path)
        # text_list.append(text_ref[idd])

    return video_paths, audio_paths, text_list



def imagebind_score(generated_video_root, ref_audio_root, prompts_path = "/mnt/data2/chris/code/Training-Free-Guidance/data/test_prompt_vgg.txt"):
    generated_video_paths = glob.glob(f"{generated_video_root}/*.mp4")
    # reference_audio_paths = []
      #pandas.read_csv("/mnt/data2/chris/code/Training-Free-Guidance/data/test_prompt_vgg.txt")

    #    image_size = (256, 256)
        # video_fps= 8
        # video_num_frame = 16
        # text_list=["A dog.", "A car", "A bird"]
        # image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
        # audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]


    # text_ref = {}
    # with open(prompts_path, "r") as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',')
    #     for prompt in reader:
    #         # print(prompt)
    #         idd, textprompt = prompt 
    #         text_ref[idd[:-2]] = textprompt.replace("\"", "")

    
    chunk_size = 32
    total_sum = 0.
    total_sum_text = 0.
    count = 0
    # Instantiate model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    
    for i in range(len(generated_video_paths)//chunk_size):
        chunk_generated_video_paths = generated_video_paths[i*chunk_size:(i+1)*chunk_size]
        video_paths, audio_paths, text_list = get_batch_paths(chunk_generated_video_paths, ref_audio_root)

        # print(">>> generated_video+_patrgyGHs: ",  text_ref["_8K8dBv9krY_000031"])

        # # Load data
        inputs = {
            ModalityType.TEXT: load_and_transform_text(text_list, device),
            ModalityType.VISION: load_and_transform_video_data(video_paths, device),
            # ModalityType.AUDIO: load_and_transform_audio_data(audio_paths, device),
        }
        
        with torch.no_grad():
            embeddings = model(inputs)
            # score = F.cosine_similarity(embeddings[ModalityType.AUDIO], embeddings[ModalityType.VISION])
            score_text = F.cosine_similarity(embeddings[ModalityType.TEXT], embeddings[ModalityType.VISION])
            count += score_text.shape[0]
            # score = score.sum()
            score_text = score_text.sum()
            
        # total_sum += score
        total_sum_text += score_text 
        print(">>score mean: ",  total_sum_text/count, count)

    print("total su m: ", total_sum_text, count) ## ours: 640.1416, device='cuda:0') 2976

    ## score: Seeing and Hearing: tensor(0.2183, device='cuda:0') tensor(0.3155, device='cuda:0') 2976

if __name__ == "__main__":

    # gt_video_root="/mnt/data2/chris/datasets/vggsound/reference_videos_2secs/"
    # generated_video_root=  "/mnt/data2/chris/outputs/video_gen_guide/dover_aes/titan_scorebased/"
    generated_video_root=  "/mnt/data2/chris/outputs/video_gen_guide/style_transfer/cogvid/titan_style_vanilla/"
    # generated_video_root = "/mnt/data2/chris/outputs/video_gen_guide/dover_aes/cogvid/titan_doveraesthetic_gradest/"
    # generated_video_root=  "/mnt/data2/chris/outputs/video_gen_guide/style_transfer/titan256x256/" #"/mnt/data2/chris/outputs/vgg/test_ours" #"/mnt/data2/chris/outputs/vgg/test_seeinghearing/" ##  ##
    ref_audio_root = "/mnt/data2/dataset/vggsound/svg_no/audio2video/audio/"
    prompts_path="data/partiprompts_1000.txt"
    imagebind_score(generated_video_root, ref_audio_root,prompts_path=prompts_path)
    # gt_video_paths = glob.glob(f"{gt_video_root}/*.mp4")
    



    
