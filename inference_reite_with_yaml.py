import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
import os
from PIL import Image
import torch.nn.functional as F
import json
from tqdm import tqdm
import sys
from pycocotools import mask as cocomask
from tools.colormap import colormap
import opts
from models.samwise import build_samwise
from util.misc import on_load_checkpoint
from tools.metrics import db_eval_boundary, db_eval_iou
from datasets.transform_utils import VideoEvalDataset
from torch.utils.data import DataLoader
from os.path import join
from datasets.transform_utils import vis_add_mask
import yaml
from moviepy import ImageSequenceClip

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_frames_from_mp4(video_path, fps, path_to_save_frames='saved_frames'):
    extract_folder = os.path.join(path_to_save_frames, os.path.basename(video_path).split('.')[0])+'_fps_'+str(fps)
    print(f'Extracting frames from .mp4 in {extract_folder} with ffmpeg...')
    if os.path.isdir(extract_folder):
        print(f'{extract_folder} already exists, using frames in that folder')
    else:
        os.makedirs(extract_folder)
        extract_cmd = "ffmpeg -i {in_path} -loglevel error -vf fps={fps} {folder}/frame_%05d.png"
        ret = os.system(extract_cmd.format(in_path=video_path, folder=extract_folder, fps=fps))
        if ret != 0:
            print('Something went wrong extracting frames with ffmpeg')
            sys.exit(ret)
    frames_list = os.listdir(extract_folder)
    frames_list = sorted([os.path.splitext(frame)[0] for frame in frames_list])
    return extract_folder, frames_list, '.png'

def compute_masks(model, text_prompt, frames_folder, frames_list, ext, device, threshold, eval_clip_window, num_workers, origin_size):
    all_pred_masks = []
    vd = VideoEvalDataset(frames_folder, frames_list, ext=ext)
    dl = DataLoader(vd, batch_size=eval_clip_window, num_workers=num_workers, shuffle=False)
    origin_w, origin_h = origin_size if origin_size else (vd.origin_w, vd.origin_h)
    
    model.eval()
    with torch.no_grad():
        for imgs, clip_frames_ids in tqdm(dl):
            clip_frames_ids = clip_frames_ids.tolist()
            imgs = imgs.to(device)
            img_h, img_w = imgs.shape[-2:]
            size = torch.as_tensor([int(img_h), int(img_w)]).to(device)
            target = {"size": size, 'frame_ids': clip_frames_ids}

            outputs = model([imgs], [text_prompt], [target])
            pred_masks = outputs["pred_masks"]
            pred_masks = pred_masks.unsqueeze(0)
            pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
            pred_masks = (pred_masks.sigmoid() > threshold)[0].cpu()
            all_pred_masks.append(pred_masks)

    all_pred_masks = torch.cat(all_pred_masks, dim=0).numpy()
    return all_pred_masks

def inference(model, config, save_path_prefix, input_path, text_prompts, resume_path):
    device = torch.device(config['model']['device'])
    threshold = config['model']['threshold']
    eval_clip_window = config['model']['eval_clip_window']
    num_workers = config['model']['num_workers']
    
    if os.path.isfile(input_path) and not config['inference']['image_level']:
        frames_folder, frames_list, ext = extract_frames_from_mp4(input_path, fps=config['inference']['fps'], 
                                                                  path_to_save_frames=config['output']['path_to_save_frames'])
        origin_size = None
    elif os.path.isfile(input_path) and config['inference']['image_level']:
        fname, ext = os.path.splitext(input_path)
        frames_list = [os.path.basename(fname)]
        frames_folder = os.path.dirname(input_path)
        origin_size = Image.open(input_path).size[::-1]  # (H, W)
    else:
        frames_folder = input_path
        frames_list = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
        ext = os.path.splitext(frames_list[0])[1] if frames_list else '.png'
        frames_list = [os.path.splitext(frame)[0] for frame in frames_list]
        origin_size = None

    print(f'---------- Starting inference on {len(frames_list)} frames ----------')
    for i, text_prompt in enumerate(text_prompts):
        all_pred_masks = compute_masks(
            model, text_prompt, frames_folder, frames_list, ext,
            device, threshold, eval_clip_window, num_workers, origin_size
        )
        video_name = input_path.split('/')[-1].split('.')[0]
        
        save_visualize_path_dir = join(save_path_prefix, 'videoname-'+ video_name + '_text_prompt-' + text_prompt.replace(' ', '_')) + '_ckpt_' + os.path.basename(resume_path).split('.')[0]
        os.makedirs(save_visualize_path_dir, exist_ok=True)
        print(f'Saving output to disk in {save_visualize_path_dir}')
        
        out_files_w_mask = []
        for t, frame in enumerate(frames_list):
            img_path = join(frames_folder, frame + ext)
            source_img = Image.open(img_path).convert('RGBA')
            source_img = vis_add_mask(source_img, all_pred_masks[t], color_list[i % len(color_list)])
            save_visualize_path = join(save_visualize_path_dir, frame + '.png')
            source_img.save(save_visualize_path)
            out_files_w_mask.append(save_visualize_path)
        
        if not config['inference']['image_level']:
            clip = ImageSequenceClip(out_files_w_mask, fps=config['inference']['fps'])
            clip.write_videofile(
                join(save_path_prefix, 'videoname-'+ video_name + '_text_prompt-' + text_prompt.replace(' ', '_') + '_ckpt_' + os.path.basename(resume_path).split('.')[0] + '.mp4'),
                codec='libx264'
            )

    print(f'Output masks and videos can be found in {save_path_prefix}')
    
def check_config(config):
    assert os.path.isfile(config['inference']['input_path']) or os.path.isdir(config['inference']['input_path']), f"The provided path {config['inference']['input_path']} does not exist"
    if os.path.isfile(config['inference']['input_path']):
        ext = os.path.splitext(config['inference']['input_path'])[1]
        assert ext in ['.jpg', '.png', '.mp4', '.jpeg'], f"Provided file extension should be one of ['.jpg', '.png', '.mp4']"
        if ext in ['.jpg', '.png', '.jpeg']: 
            config['inference']['image_level'] = True
            pretrained_model = 'pretrain/pretrained_model.pth'
            pretrained_model_link = 'https://drive.google.com/file/d/1gRGzARDjIisZ3PnCW77Y9TMM_SbV8aaa/view?usp=drive_link'
            print(f'Specified path is an image, using image-level configuration')

    if not config['inference']['image_level']: # it's video inference
        # set default args
        config['model']['HSA'] = True
        config['model']['use_cme_head'] = False
        pretrained_model = 'pretrain/final_model_mevis.pth'
        pretrained_model_link = 'https://drive.google.com/file/d/1Molt2up2bP41ekeczXWQU-LWTskKJOV2/view?usp=sharing'
        print(f'Specified path is a video or folder with frames, using video-level configuration')
        
    if config['model']['resume'] == '':
        config['model']['resume'] = pretrained_model

    assert os.path.isfile(config['model']['resume']), f"You should download the model checkpoint first. Run 'cd pretrain &&  gdown --fuzzy {pretrained_model_link}"

def main(config_path='config.yaml'):
    config = load_config(config_path)
    check_config(config)
    
    # Fix seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Model
    model = build_samwise(argparse.Namespace(**config['model']))
    device = torch.device(config['model']['device'])
    model.to(device)

    resume_path = config['model'].get('resume')
    if resume_path:
        print('Loading checkpoint from {}'.format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        if list(checkpoint['model'].keys())[0].startswith('module'):
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}        
        checkpoint = on_load_checkpoint(model, checkpoint)
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        print('No checkpoint provided, using random weights')

    # Save path
    save_path_prefix = config['output']['save_path_prefix']
    os.makedirs(save_path_prefix, exist_ok=True)

    start_time = time.time()
    print('---------- Starting inference ----------')
    inference(
        model,
        config,
        save_path_prefix,
        config['inference']['input_path'],
        config['inference']['text_prompts'],
        resume_path
    )
    end_time = time.time()
    total_time = end_time - start_time
    print("---------- Total inference time: %.4f s ----------" % total_time)

if __name__ == '__main__':
    main()
