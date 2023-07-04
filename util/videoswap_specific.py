import os 
import time
import glob
import shutil
import numpy as np
from tqdm import tqdm

import imageio_ffmpeg
import torch
import torch.nn.functional as F
from moviepy.editor import VideoFileClip 

from util.reverse2original import reverse2wholeimage
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def norm_embed(embed, axis=-1):
    return embed * (1 / np.linalg.norm(embed, axis=axis, keepdims=True))

def getCosSim(src_embed, ref_embed, normed=False):
    # cosine similarity
    if not normed:
        src_embed = norm_embed(src_embed)
        ref_embed = norm_embed(ref_embed)
    return np.dot(src_embed, ref_embed)

def img_to_tensor(img_array):
    img_array = img_array.transpose(-1, -3, -2) # HWC => CHW
    img_array = img_array.astype(np.float32) * (1 / 255) # [0,255] => [0,1]
    img_tensor = torch.from_numpy(img_array)
    return img_tensor

def video_swap(video_path, id_vector, specific_person_id_nonorm, id_thres, swap_model, detect_model, save_path,
    temp_results_dir='./temp_results', crop_size=224, no_simswaplogo=False,use_mask=False):
    # audio checker
    # video_forcheck = VideoFileClip(video_path)
    # no_audio = video_forcheck.audio is None
    # del video_forcheck

    logoclass = None if no_simswaplogo else watermark_image('./simswaplogo/simswaplogo.png')

    spNorm = SpecificNorm()
    mse = torch.nn.MSELoss().cuda()

    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net = None

    # initialize input video reader
    src_reader = imageio_ffmpeg.read_frames(video_path, pix_fmt='rgb24')
    src_meta = next(src_reader)
    src_fps = src_meta['fps']
    src_size = src_meta['size']
    src_width, src_height = src_size
    est_frames = src_meta['duration'] * src_fps
    print(f'Metadata:\n{src_meta}')
    print(f'Estimated number of frames: {est_frames}')

    # initialize output video writer
    dst_writer = imageio_ffmpeg.write_frames(save_path, src_size, pix_fmt_in='rgb24',
        fps=src_fps, quality=6, macro_block_size=1, audio_path=video_path)
    dst_writer.send(None)

    # try-finally to handle release of the video reader and video writer
    try:
        # loop over the frames
        for src_frame in tqdm(src_reader):
            # convert RGB24 data from bytes to image array
            src_frame = np.frombuffer(src_frame, dtype=np.uint8)
            src_frame = src_frame.reshape(src_height, src_width, 3)

            # face detection
            detect_results = detect_model.get(src_frame, crop_size)

            # face swap
            if detect_results is not None:
                frame_align_crop_list = detect_results[0]
                frame_mat_list = detect_results[1]
                id_compare_values = []
                frame_align_crop_tensor_list = []
                for frame_align_crop in frame_align_crop_list:
                    frame_align_crop_tensor = img_to_tensor(frame_align_crop)[None, ...].cuda()
                    frame_align_crop_tensor_list.append(frame_align_crop_tensor)
                    frame_align_crop_tensor_arcnorm = spNorm(frame_align_crop_tensor)
                    frame_align_crop_tensor_arcnorm_downsample = F.interpolate(frame_align_crop_tensor_arcnorm, size=(112, 112))
                    frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tensor_arcnorm_downsample)
                    id_distance = 1 - getCosSim(
                        frame_align_crop_crop_id_nonorm.squeeze(0).detach().cpu().numpy(),
                        specific_person_id_nonorm.squeeze(0).detach().cpu().numpy(),
                    )
                    id_compare_values.append(id_distance)
                id_compare_values_array = np.array(id_compare_values)
                min_index = np.argmin(id_compare_values_array)
                min_value = id_compare_values_array[min_index]
                if min_value < id_thres:
                    _frame_align_crop_tensor = frame_align_crop_tensor_list[min_index]
                    swap_result = swap_model(None, _frame_align_crop_tensor, id_vector, None, True)[0]
                    swap_result_list = [swap_result]
                    frame_save_path = None
                    dst_frame = reverse2wholeimage(
                        [_frame_align_crop_tensor], swap_result_list, [frame_mat_list[min_index]],
                        crop_size, src_frame, frame_save_path,
                        pasring_model=net, use_mask=use_mask, norm=spNorm
                    )
                else:
                    dst_frame = src_frame
            else:
                dst_frame = src_frame

            # write frame to output video
            if not no_simswaplogo:
                dst_frame = logoclass.apply_frames(dst_frame)
            dst_writer.send(dst_frame)
    finally:
        # release
        src_reader.close()
        dst_writer.close()
