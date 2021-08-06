import torch
import math
from noise import pnoise2

def normalize(img):
    norm_img = img.clone()
    return (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min() + 1e-30)

def contrast(img, freq):
    return torch.sin(img * math.pi * freq)

def noise_generator(d1, d2, period_x, period_y, octave, freq, lacunarity=2):

    period_x = period_x.item()
    period_y = period_y.item()
    octave = int(round(octave.item()))
    freq = freq.item()

    ###########################
    noise = torch.empty((d1, d2), dtype=torch.double)
    for x in range(d1):
        for y in range(d2):
            noise[x,y] = pnoise2(x / period_x, y / period_y, octaves=octave, lacunarity = lacunarity)

    return contrast(normalize(noise), freq=freq)



# start = start.item()
# octaves = octaves.item()
# freq = freq.item()

# if end == None:
#     end = start + 1
# axis1 = torch.linspace(start,end,d1)
# axis2 = torch.linspace(start,end,d2)
# mesh = torch.meshgrid(axis1, axis2)
# ts_noise = torch.zeros_like(mesh[0])
# octaves_round = int(round(octaves))
# for i in range(d1):
#     for j in range(d2):
#         ts_noise[i,j] = pnoise2(mesh[0][i,j], mesh[1][i,j], octaves=octaves_round)


    ###########################