import os
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from video_process import PointSet
from glob import glob
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2_video_predictor

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
video_predicator = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


class InterferenceFrame:

    def __init__(self):        
        self.origin_frame_id = 0
        self.item_id = 0
        self.point_set = None


def prepare_path(base_path):
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    results = []
    results.append(os.path.join(base_path, date_str, time_str, 'origin'))
    results.append(os.path.join(base_path, date_str, time_str, 'mask'))
    results.append(os.path.join(base_path, date_str, time_str, 'result'))
    for d in results:
        os.makedirs(d, exist_ok = True)
    results.append(os.path.join(base_path, date_str, f'{time_str}.mp4'))
    return results


def preprare_video_frames(source_video, frame_path):
    command = f'ffmpeg -i {source_video} -q:v 2 -start_number 0 {os.path.join(frame_path, "%05d.jpg")} 2>&1'
    with os.popen(command) as fp:
        fp.readlines()


def merge_mask(origin_file, mask_file, result_file):
    origin_file = os.path.abspath(origin_file)
    mask_file = os.path.abspath(mask_file)
    result_file = os.path.abspath(result_file)
    command = f'ffmpeg -i {mask_file} -i {origin_file} -filter_complex "[0][1]blend=all_expr=0.3*A+0.7*B" {result_file} 2>&1 '
    print(f'command = {command}')
    with os.popen(command) as fp:
        fp.readlines()

def merge_video(result_dir, result_file):
    command = f'ffmpeg -f image2 -i {os.path.join(result_dir, "%05d.jpg")} {result_file}'
    with os.popen(command) as fp:
        fp.readlines()


def video_interfrence(video_path, output_path, frames, width, height):
    # prepare pathes
    origin_path, mask_path, result_path, final_file = prepare_path(output_path)
    # extract frames from video
    preprare_video_frames(video_path, origin_path)
    # initialize predicator
    inference_state = video_predicator.init_state(video_path=origin_path)
    for f in frames:
        params = {
            'inference_state': inference_state,
            'frame_idx': f.origin_frame_id,
            'obj_id': f.item_id,
            'points': [(x[0], x[1]) for x in f.point_set],
            'labels': [ x[2] for x in f.point_set]
        }
        video_predicator.add_new_points_or_box(**params)

    video_segments = {}  # video_segments contains the per-frame segmentation results
    # propagation
    for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(video_predicator.propagate_in_video(inference_state), desc='video propagation'):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # combine the result and visualization
    color_mask_dict = {}
    for out_frame_idx in tqdm(range(len(video_segments)), desc='merge masks'):
        mask_all = np.ones((width, height, 3))
        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                if out_obj_id not in color_mask_dict:
                    color_mask_dict[out_obj_id] = np.random.random((1, 3)).tolist()[0]
                color_mask = color_mask_dict[out_obj_id]
                for i in range(3):
                    mask_all[out_mask[0] == True, i] = color_mask[i]
        img = Image.fromarray(np.uint8(mask_all * 255)).convert('RGB')
        file_name = '%05d.jpg' % out_frame_idx
        full_file_name = os.path.join(mask_path, file_name)
        full_origin_name = os.path.join(origin_path, file_name)
        full_result_name = os.path.join(result_path, file_name)
        img.save(full_file_name, format='JPEG')
        if os.path.exists(full_origin_name):
            merge_mask(full_origin_name, full_file_name, full_result_name)
    merge_video(result_path, final_file)
    torch.cuda.empty_cache()
    gc.collect()
    return final_file
        
            
                
            
            