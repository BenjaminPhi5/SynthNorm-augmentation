from torchio.transforms import (
    RandomMotion,
    RandomGhosting,
    RandomSpike,
    RandomBlur,
    RandomAnisotropy,
)

import numpy as np
import torch
from twaibrain.braintorch.augmentation.perlin_noise import perlin_improved_noise_gpu

def apply_bias_field(I, B, mask=None):
    """
    I = image 
    B = bias field
    mask = brainmask for applying zscore normalization. if mask is not provided here
    this must be done somewhere else.

    shift the image to have 0 min, then apply bias field and then re-zscore normalize the image.

    bias field correction should be done as the first augmentation
    """
    minv = I.min()
    Ib = (I - (minv * 2)) * B + (minv * 2) # take minv * 2 so that parts of the image may get darker than the darkest part of the original image once rescaled.
    
    if mask != None:
        Ib = Ib * mask
    
    return Ib


class PerlinBiasField:
    def __init__(self, shape=[40, 96, 80], base_res=[20, 24, 20], res_scales=[1,2,2,3,3,3,4,4,4], device='cpu', min_range=(0.4, 1), max_range=(1, 1.6), return_field=False, mask_at_0=False, resample=True, resample_factor=2):
        """
        stochastic perlin bias field generation function

        shape: the shape of the bias field that is generated.
        base_res: the distance in voxels between grid coordinates used in the perlin noise algorithm
        res_scales: list of scales used to multiple the base_res. a single scale is randomly sampled from the list.
        device: cpu or cuda, torch device used for the compute. gpu preferable for speed.
        min_range: range of minimum values for the resulting bias field, sampled from uniformly
        max_range: range of maximum values for the resulting bias field, sampled from uniformly.

        note: mask_at_0 is used here only for visualization processes, during training mask multiplication is done later in the augmentation pipeline, see ArtefactsAugmentation class
        """
        
        self.shape = shape
        self.base_res = np.array(base_res)
        self.res_scales = res_scales
        self.min_range = min_range
        self.max_range = max_range
        self.device = device
        self.return_field = return_field
        self.mask_at_0 = mask_at_0
        self.resample = resample
        self.resample_factor = resample_factor

    def __call__(self, img):
        min_value = np.random.uniform(low=self.min_range[0], high=self.min_range[1])
        max_value = np.random.uniform(low=self.max_range[0], high=self.max_range[1])
        res = self.base_res * np.random.choice(self.res_scales)
        mask = None if not self.mask_at_0 else img != 0
        
        B = perlin_improved_noise_gpu(self.shape, res, self.device)
    
        B = B - B.min()
        B = B / B.max()
        
        B = B * (np.log(max_value) - np.log(min_value))
        B = B + np.log(min_value)
        
        B = B.exp().unsqueeze(0).to(img.device)

        if self.resample:
            B = torch.nn.functional.interpolate(B.unsqueeze(0), scale_factor=self.resample_factor, mode='trilinear').squeeze(0)

        if self.return_field:
            return apply_bias_field(img, B, mask), B

        return apply_bias_field(img, B, mask)


    def testing(self):
        min_value = np.random.uniform(low=self.min_range[0], high=self.min_range[1])
        max_value = np.random.uniform(low=self.max_range[0], high=self.max_range[1])
        res = self.base_res * np.random.choice(self.res_scales)
        
        B = perlin_improved_noise_gpu(self.shape, res, self.device)
        R = B.clone()
        R = (R - R.min()) / (R.max() - R.min())
        R = R * (max_value - min_value) + min_value
    
        B = B - B.min()
        B = B / B.max()
        
        B = B * (np.log(max_value) - np.log(min_value))
        B = B + np.log(min_value)
        
        B = B.exp().unsqueeze(0)
        R = R.unsqueeze(0)

        if self.resample:
            B = torch.nn.functional.interpolate(B.unsqueeze(0), scale_factor=self.resample_factor, mode='trilinear').squeeze(0)
            R = torch.nn.functional.interpolate(R.unsqueeze(0), scale_factor=self.resample_factor, mode='trilinear').squeeze(0)

        return B, R


class ArtefactsAugmentation:
    def __init__(self, keys=['FLAIR'], p_bias_field=0.5, p_downsample=0.1, p_motion=0.05, p_ghosting=0.05, p_spike=0.05, modality_p=1.0, bias_field='perlin'):
        self.keys = keys
        self.modality_p = modality_p
        bias_field_aug = PerlinBiasField() if bias_field=='perlin' else SynthsegBiasField()
        self.transforms = [
            # [p_bias_field, RandomBiasField(coefficients=0.75, order=2)], #  yes happy with this
            [p_bias_field, bias_field_aug],
            [p_spike, RandomSpike(num_spikes=(1,1), intensity=5)],
            [p_motion, RandomMotion(degrees=2, translation=2, num_transforms=3)],
            [p_ghosting, RandomGhosting(num_ghosts=(1,5), axes=(0,1,2), intensity=(0.5, 1), restore=(0,0.5))],
            [p_downsample, RandomAnisotropy(axes=(0), downsampling=(1, 1.5))],
            [p_downsample, RandomAnisotropy(axes=(1), downsampling=(1.25, 2.5))],
            [p_downsample, RandomAnisotropy(axes=(2), downsampling=(1.25, 2.5))],
        ]
    
    def __call__(self, data):
        for key in self.keys:
            inp = data[key]
            maxv = inp.max().item()
            minv = inp.min().item()
            modified = False
            for (p_transform, transform) in self.transforms:
                r = np.random.rand()
                if r < p_transform and r < self.modality_p:
                    modified = True
                    inp = transform(inp)

            if modified:
                inp = inp * (data['brainmask'] == 1)
                inp = inp.clamp(minv-1, maxv+1)
            data[key] = inp

        return data


