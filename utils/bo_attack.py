import torch
import numpy as np

from utils.perlin import noise_generator


def gen_noise(img, period_x, period_y, octaves, freq):
    # noise_img = noise_generator(img.shape[0], img.shape[1], octaves=octaves, freq=freq, start=start).unsqueeze(-1)
    noise_img = noise_generator(*img.shape[-2:], period_x=period_x, period_y=period_y, octave=octaves, freq=freq).unsqueeze(-1)
    return noise_img

def add_noise(img, noise, norm): 
    img = img + norm * noise
    return np.clip(img, 0, 255)

def evaluate(img, detector, model, period_x, period_y, octaves, freq, imgsz=640, norm=16):
    '''
    Evaluate the obj value with given parameters
    '''
    batch_obj_conf = torch.zeros(octaves.numel(), 1)
    batch_pred = []     

    for i in range(octaves.numel()):
        # noise = gen_noise(img.transpose(1,2,0), octaves[i], freq[i], start[i])
        noise = gen_noise(img, period_x[i], period_y[i], octaves[i], freq[i])
        perturb_img = add_noise(img=img, noise=noise.expand(*noise.size()[:2], 3).numpy().transpose(2,0,1), norm=norm)
        pred, loss= detector(perturb_img, model, img_size=imgsz)
        batch_obj_conf[i,0] = loss[0][0] +loss[1][0] + loss[2][0]
        batch_pred.append(pred[0])
    return batch_pred, batch_obj_conf
