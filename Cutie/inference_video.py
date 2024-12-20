import copy
import os
import time

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model,get_lightweight_model
import cv2

@torch.inference_mode()
@torch.cuda.amp.autocast()
def initialize_processor():
    #cutie = get_default_model()
    cutie = get_lightweight_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    processor.max_internal_size = 480

    return processor

@torch.inference_mode()
@torch.cuda.amp.autocast()
def inference_video(processor, input_video_path, input_first_mask_path, output_video_path):
    mask = Image.open(input_first_mask_path)
    assert mask.mode in ['L', 'P']
    palette = mask.getpalette()


    objects = np.unique(np.array(mask))
    objects = objects[objects != 0].tolist()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = torch.from_numpy(np.array(mask)).to(device)

    video_reader_cap = cv2.VideoCapture(input_video_path)

    frame_width = int(video_reader_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video_reader_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_reader_cap.get(cv2.CAP_PROP_FPS)

    video_writer_cap = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    current_frame_index = 0
    while video_reader_cap.isOpened():
        ret, frame = video_reader_cap.read()
        if not ret:
            break

        #print(f'Processing frame: {current_frame_index} / {frame_count}')

        original_frame = copy.deepcopy(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_array = np.array(rgb_frame)
        frame = to_tensor(frame_array)
        frame = frame.to(device).float()

        start_time = time.perf_counter()

        if current_frame_index == 0:
            output_prob = processor.step(frame, mask, objects=objects)
        else:
            output_prob = processor.step(frame)

        result_mask = processor.output_prob_to_mask(output_prob)
        result_mask = result_mask.cpu().numpy().astype(np.uint8)

        end_time = time.perf_counter()
        #print(f'Inference time: {(end_time - start_time) * 1000:.4f} ms')

        # 显示方式1：黑白图
        #result_mask = (result_mask * 255).astype(np.uint8)
        #result_mask_image = cv2.cvtColor(result_mask, cv2.COLOR_RGB2BGR)

        # 叠加到原图上
        #result_mask_image = cv2.addWeighted(original_frame, 0.5, result_mask_image, 0.5, 0)

        # 显示方式2：RGB
        result_mask_image = Image.fromarray(result_mask)
        result_mask_image.putpalette(palette)
        result_mask_image = result_mask_image.convert('RGB')
        result_mask_image = cv2.cvtColor(np.asarray(result_mask_image), cv2.COLOR_RGB2BGR)
        # 叠加到原图上
        result_mask_image = cv2.addWeighted(original_frame, 0.4, result_mask_image, 0.6, 0)

        video_writer_cap.write(result_mask_image)

        current_frame_index += 1

    video_reader_cap.release()
    video_writer_cap.release()

    print(f'Video saved to {output_video_path}')



if __name__ == '__main__':
    input_video_path = './inference_test/input_videos/测试集/0.23_1280x720_AVC_25fps.m4v'
    input_first_mask_path = './inference_test/input_videos/测试集/0.23_1280x720_AVC_25fps_mask.png'
    output_root_dir = './inference_test/output_videos/测试集'

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    input_video_name = os.path.basename(input_video_path).rsplit('.', 1)[0]
    output_video_dir = os.path.join(output_root_dir, input_video_name)
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)

    output_video_path = os.path.join(output_video_dir, f'{input_video_name}.mp4')

    processor = initialize_processor()

    inference_video(processor, input_video_path, input_first_mask_path, output_video_path)