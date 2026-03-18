from monai.transforms import MapTransform
from abc import abstractmethod
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import numpy as np
from monai.transforms import RandRotated
from monai.transforms import RandAffined
from monai.transforms import RandGaussianNoised
from monai.transforms import RandGaussianSmooth
from monai.transforms import RandScaleIntensityd
from monai.transforms import RandAdjustContrastd
from monai.transforms import RandFlipd
from monai.transforms import Compose
from torch import float32, long, cat
import torch.nn.functional as F
from synthnormaug.artefacts_augmentation import ArtefactsAugmentation
from synthnormaug.synthetic_intensity_augmentation.SyntheticGMMAugmentation import SyntheticGMMAugmentation
from synthnormaug.label_format import OneVRest, OneHotEncoder

import torch
import numpy as np
from scipy.ndimage import zoom


class MonaiAugmentationExtender(MapTransform):
    def __init__(self, p:float, dims:int=2, keys=['image', 'mask', 'label'], mode=None, *args, **kwargs):
        super().__init__(keys=keys, *args, **kwargs)
        assert dims == 2 or dims == 3
        assert 0 <= p <= 1
        self.dims = dims # 2 for 2D, 3 for 3D
        self.mode= mode # modes are used where we treat labels and images differently.
        self.p = p # p is probability of applying the transform.

    @abstractmethod
    def __call__(self, data):
        pass

class MonaiAugmentationWrapper(MonaiAugmentationExtender):
    # needs to have base_augmentation defined in the constructor
    def __call__(self, data):
        return self.base_augmentation(data)


class RotationAugment(MonaiAugmentationWrapper):
    def __init__(self, axial_only=False, *args, **kwargs):
        super().__init__(*args, mode=['bilinear', 'nearest', 'nearest'], **kwargs)

        pi = np.pi
        
        degrees_3D=(-30 * pi/180,30 * pi/180)
        degrees_2D =(-pi,pi)

        if axial_only and self.dims == 3:
            range_x = range_y = (0,0)
            range_z = degrees_2D

        elif self.dims == 3:
            range_x = range_y = range_z = degrees_3D

        elif self.dims == 2:
            range_x = range_y = range_z = degrees_2D

        self.base_augmentation = RandRotated(
            keys=self.keys,
            mode=self.mode,
            range_x=range_x,
            range_y=range_y,
            range_z=range_z,
            keep_size=True,
            prob=self.p
        )


class AffineAugment(MonaiAugmentationExtender):
    def __init__(self, spatial_size, axial_only=False, *args, mode=['bilinear', 'nearest', 'nearest'], **kwargs):
        """
        spatial size should be the output size I want I think? (or just the size of the data
        , i.e dont change the size
        I should do the affine translation and then centre crop perhaps? I'm not really sure how the
        spatial dim stuff works at the moment, and when I try 3D I should think about this.
        """
        super().__init__(*args, mode=['bilinear', 'nearest', 'nearest'], **kwargs)
        self.spatial_size = spatial_size

        pi = np.pi

        ### setup rotation range
        # degrees_3D=(-30 * pi/180,30 * pi/180) # for isotropic
        degrees_2D = (-pi,pi)
        degrees_3D = degrees_2D # my data is anisotropic 3D for now.

        if axial_only and self.dims == 3:
            range_x = range_z = (0,0)
            range_y = degrees_2D

        elif self.dims == 3:
            range_x = range_y = range_z = degrees_3D

        elif self.dims == 2:
            range_x = range_y = range_z = degrees_2D

        self.rotate_range = (*range_x, *range_y, *range_z)

        ### setup translation range
        translation_scale = 0.1
        self.translate_range = [(-s * translation_scale, s * translation_scale) for s in spatial_size]

        ### setup scale range
        scale_min = -0.3
        scale_max = 0.4
        self.scale_range = (scale_min, scale_max, scale_min, scale_max, scale_min, scale_max)

        ### setup shear range
        shear_angle = 18 * pi/180
        if axial_only and self.dims == 3:
            self.shear_range = (0, 0, -shear_angle, shear_angle, 0,0)
        else:
            self.shear_range = (-shear_angle, shear_angle, -shear_angle, shear_angle, -shear_angle)
        

    def __call__(self, data):
        # decide whether to call augmentation:
        do_rotation = np.random.uniform(0,1) < self.p 
        do_scale = np.random.uniform(0,1) < self.p
        do_translate = np.random.uniform(0,1) < self.p
        do_shear = np.random.uniform(0,1) < self.p

        if not do_rotation and not do_scale and not do_translate and not do_shear:
            return data

        rotate_range = self.rotate_range if do_rotation else None
        # print(rotate_range)
        scale_range = self.scale_range if do_scale else None
        translate_range = self.translate_range if do_translate else None
        shear_range = self.shear_range if do_shear else None
        
        augment = RandAffined(
            keys=self.keys,
            mode=self.mode,
            spatial_size=self.spatial_size, 
            prob=1,
            rotate_range=rotate_range,
            scale_range=scale_range,
            translate_range=translate_range,
            shear_range=shear_range,
            padding_mode='zeros'
        )
        
        return augment(data)

class GaussianNoiseAugment(MonaiAugmentationWrapper):
    def __init__(self, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.base_augmentation = RandGaussianNoised(
            keys=self.keys,
            prob=self.p,
            mean=0,
            std=0.1 # this parameter is the top end of the rane that the noise is sampled from (I think)
        )


class GaussianBlurAugment(MonaiAugmentationExtender):
    def __init__(self, modality_p, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.modality_p = modality_p
        self.base_augmentation = RandGaussianSmooth(
            prob=self.modality_p,
            sigma_x=(0.5,1.5),
            sigma_y=(0.5,1.5),
            sigma_z=(0.5,1.5),
        )
        # print(self.p)
        # print(self.modality_p)

    def __call__(self, data):
        if np.random.uniform(0,1) > self.p:
            # print("returning")
            return data

        for key in self.keys:
            key_data = data[key].clone()
            for channel in range(key_data.shape[0]):
                channel_data = key_data[channel].unsqueeze(0)
                if np.random.uniform(0,1) < self.modality_p:
                    channel_data = self.base_augmentation(channel_data)
                key_data[channel] = channel_data

            data[key] = key_data
        return data

class BrightnessAugment(MonaiAugmentationWrapper):
    def __init__(self, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        # monai uses v = v * (1 + factor)
        # I want to multiply by range 0.7, to 1.3,, so in range -0.3, to 0.3
        self.base_augmentation = RandScaleIntensityd(
            keys=self.keys,
            factors=(-0.3,0.3),
            prob=self.p,
        )

class ContrastAugment(MonaiAugmentationWrapper):
    def __init__(self, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.base_augmentation = RandAdjustContrastd(
            keys=self.keys,
            prob=self.p,
            gamma=(0.65,1.5)
        )


class LowResolutionSimulationAugmentation(MonaiAugmentationExtender):
    def __init__(self, modality_p, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.modality_p = modality_p

    def __call__(self, data):
        # decide whether to call augmentation:
        if np.random.uniform(0,1) > self.p:
            return data

        # generate downsample factor
        factor = np.random.uniform(0.5, 1)
        # print(factor)

        for key in self.keys:
            if key not in data:
                raise KeyError(f"key {key} not found and allow_missing_keys==False for this augmentation")

            key_data = data[key].clone()
            for channel in range(key_data.shape[0]):
                if np.random.uniform(0,1) < self.modality_p:
                    # print(key_data[channel].shape)
                    channel_data = key_data[channel].unsqueeze(0)
                    spatial_dims = np.array(channel_data.shape[-2:]) # only take the last two dims, so don't downsample the already low resolution z plane.
                    downsampled_size = np.round(spatial_dims * factor).astype(np.int32)
                    # print(downsampled_size)
                    downsampled = resize(channel_data, downsampled_size, interpolation=InterpolationMode.NEAREST)
                    upsampled = resize(downsampled, spatial_dims, interpolation=InterpolationMode.BICUBIC, antialias=True)
                    # print(upsampled.squeeze().shape)
                    key_data[channel] = upsampled.squeeze()
            data[key] = key_data
            
        return data


class GammaAugmentation(MonaiAugmentationExtender):
    def __init__(self, *args, keys=['image'], allow_invert=True, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.allow_invert=allow_invert
    
    def __call__(self, data):
        # decide whether to call augmentation:
        if np.random.uniform(0,1) > self.p:
            return data

        gamma = np.random.uniform(0.7, 1.5)

        invert = self.allow_invert and np.random.uniform(0,1) < 0.15

        for key in self.keys:
            if key not in data:
                raise KeyError(f"key {key} not found and allow_missing_keys==False for this augmentation")

            key_data = data[key]

            # 0, 1 scale
            drange = (key_data.min(), key_data.max())
            key_data = (key_data - drange[0]) / (drange[1] - drange[0])

            # gamma scale
            if invert:
                key_data = 1 - (1-key_data).pow(gamma)
            else:
                key_data = key_data.pow(gamma)

            # original range scale
            key_data = key_data * (drange[1] -drange[0]) + drange[0]
            
            data[key] = key_data

        return data

class MirrorAugment(MonaiAugmentationExtender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flip0 = RandFlipd(keys=self.keys, prob=self.p, spatial_axis=0)
        self.flip1 = RandFlipd(keys=self.keys, prob=self.p, spatial_axis=1)
        self.flip2 = RandFlipd(keys=self.keys, prob=self.p, spatial_axis=2)

    def __call__(self, data):
        # todo: is this nessesary? what happens if I ignore this?
        if self.dims == 2:
            return self.flip0(data)
        else:
            return self.flip0(self.flip1(self.flip2(data)))


class SetDtype():
    def __init__(self, keys, dtypes):
        self.keys = keys
        self.dtypes = dtypes

    def __call__(self, data):
        for i, key in enumerate(self.keys):
            data[key] = data[key].type(self.dtypes[i])

        return data
        

class SetDtypeImageLabelPair():
    def __init__(self, keys, dtypes):
        self.keys = keys
        self.dtypes = dtypes

    def __call__(self, data):
        for i, key in enumerate(self.keys):
            data[key] = data[key].type(self.dtypes[i])

        return cat([data['image'], data['mask']]), data['label']


class MonaiPairedPadToShape2d:
    def __init__(self, target_shape, padding_mode="constant", padding_value=0):
        self.target_shape = target_shape
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def __call__(self, data):
        img = cat([data['image'], data['mask']])
        label = data['label']
        
        _, h, w = img.shape
        target_h, target_w = self.target_shape

        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        img = F.pad(img, padding, mode=self.padding_mode, value=self.padding_value)
        label = F.pad(label, padding, mode=self.padding_mode, value=self.padding_value)

        data = {
            'image':img[0:2],
            'mask':img[2].unsqueeze(0),
            'label':label
        }
        
        return data

class MonaiPairedPadToShape3d:
    def __init__(self, target_shape, padding_mode="constant", padding_value=0):
        self.target_shape = target_shape
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def __call__(self, data):
        img = cat([data['image'], data['mask']])
        label = data['label']
        
        _, d, h, w = img.shape
        target_d, target_h, target_w = self.target_shape

        pad_d = max(0, target_d - d)
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2)
        img = F.pad(img, padding, mode=self.padding_mode, value=self.padding_value)
        label = F.pad(label, padding, mode=self.padding_mode, value=self.padding_value)

        data = {
            'image':img[0:2],
            'mask':img[2].unsqueeze(0),
            'label':label
        }
        
        return data

class MonaiCropAndPadToShape3d:
    def __init__(self, target_shape, padding_mode="constant", padding_value=0):
        self.target_shape = target_shape
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def __call__(self, data):
        img = data['image']
        mask = data['mask']
        label = data['label']
        
        _, d, h, w = img.shape
        target_d, target_h, target_w = self.target_shape

        crop_d = max(0, d - target_d)
        crop_h = max(0, h - target_h)
        crop_w = max(0, w - target_w)
        
        pad_d = max(0, target_d - d)
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        cropping = (crop_d // 2, crop_d - crop_d // 2, crop_h // 2, crop_h - crop_h // 2, crop_w, crop_w - crop_w // 2)
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2)
        
        img = img[:,cropping[0]:d-cropping[1],cropping[2]:h-cropping[3], cropping[4]:w-cropping[5]]
        mask = mask[:,cropping[0]:d-cropping[1],cropping[2]:h-cropping[3], cropping[4]:w-cropping[5]]
        label = label[:,cropping[0]:d-cropping[1],cropping[2]:h-cropping[3], cropping[4]:w-cropping[5]]
        
        img = F.pad(img, padding, mode=self.padding_mode, value=self.padding_value)
        mask = F.pad(mask, padding, mode=self.padding_mode, value=self.padding_value)
        label = F.pad(label, padding, mode=self.padding_mode, value=self.padding_value)

        data = {
            'image':img,
            'mask':mask,
            'label':label
        }
        
        return data

        
def tricubic_resize(volume: torch.Tensor, size: tuple):
    # volume: (D, H, W)
    vol_np = volume.cpu().numpy()
    size = np.array(size)
    scale_factor = size / np.array(vol_np.shape)
    resized = zoom(vol_np, zoom=scale_factor, output=np.zeros(size), order=3)  # order=3 = cubic
    
    return torch.from_numpy(resized).to(volume.device)

class LowResolutionSimulationAugmentation_V2(MonaiAugmentationExtender):
    def __init__(self, modality_p, axial_only=False, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.modality_p = modality_p
        self.axial_only = axial_only

    def __call__(self, data):
        # decide whether to call augmentation:
        if np.random.uniform(0,1) > self.p:
            return data

        # generate downsample factor
        factor = np.random.uniform(0.5, 1)
        # print(factor)

        for key in self.keys:
            if key not in data:
                raise KeyError(f"key {key} not found and allow_missing_keys==False for this augmentation")

            key_data = data[key].clone()
            if np.random.uniform(0,1) > self.modality_p:
                continue
            spatial_dims = key_data.shape[-3:] if len(key_data.shape) == 4 else key_data.shape[-2:]
            spatial_dims = np.array(spatial_dims)
            downsampled_size = np.round(spatial_dims * factor).astype(np.int32)
            if self.axial_only and len(spatial_dims) == 3:
                downsampled_size[0] = spatial_dims[0]
            
            # print(key_data.shape)
            downsampled = F.interpolate(key_data.unsqueeze(0), tuple(downsampled_size), mode='nearest')
            # print(downsampled.shape)
            # upsampled = F.interpolate(downsampled, tuple(spatial_dims), mode='bicubic', antialias=False)
            upsampled = tricubic_resize(downsampled.squeeze(), spatial_dims)
            key_data = upsampled.unsqueeze(0)
            # print(key_data.shape)
            data[key] = key_data
            
        return data

class AffineAugment_V2(MonaiAugmentationExtender):
    def __init__(self, spatial_size, axial_only=True, allow_translate=True, allow_shear=True, *args, mode=['bilinear', 'nearest', 'nearest'], **kwargs):
        """
        spatial size should be the output size I want I think? (or just the size of the data
        , i.e dont change the size
        I should do the affine translation and then centre crop perhaps? I'm not really sure how the
        spatial dim stuff works at the moment, and when I try 3D I should think about this.
        """
        super().__init__(*args, mode=['bilinear', 'nearest', 'nearest'], **kwargs)
        self.spatial_size = spatial_size

        pi = np.pi

        ### setup rotation range
        degrees_3D = (-30 * pi/180, 30 * pi/180) # for isotropic
        degrees_2D = (-pi,pi)

        if axial_only and self.dims == 3:
            range_x = range_y = (0,0)
            range_z = degrees_2D

        elif self.dims == 3:
            range_x = degrees_3D
            # range_x = (0, 0)

            range_y = degrees_3D
            # range_y = (0, 0)

            range_z = degrees_3D
            # range_z = (0, 0)
        
        elif self.dims == 2:
            range_x = range_y = range_z = degrees_2D

        self.rotate_range = (range_z, range_x, range_y)

        ### setup translation range
        translation_scale = 0.1
        self.translate_range = [(-s * translation_scale, s * translation_scale) for s in spatial_size]

        ### setup scale range
        scale_min = -0.3
        scale_max = 0.4
        self.scale_range = (scale_min, scale_max, scale_min, scale_max, scale_min, scale_max)

        ### setup shear range
        shear_angle = 18 * pi/180
        self.shear_range = (shear_angle, shear_angle, shear_angle, shear_angle, shear_angle, shear_angle)

        self.allow_shear = allow_shear
        self.allow_translate = allow_translate
        

    def __call__(self, data):
        # decide whether to call augmentation:
        do_rotation = np.random.uniform(0,1) < self.p 
        do_scale =  np.random.uniform(0,1) < self.p
        do_translate = np.random.uniform(0,1) < self.p and self.allow_translate
        do_shear = np.random.uniform(0,1) < self.p and self.allow_shear

        if not do_rotation and not do_scale and not do_translate and not do_shear:
            return data

        rotate_range = self.rotate_range if do_rotation else None
        scale_range = self.scale_range if do_scale else None
        translate_range = self.translate_range if do_translate else None
        shear_range = self.shear_range if do_shear else None
        
        augment = RandAffined(
            keys=self.keys,
            mode=self.mode,
            spatial_size=self.spatial_size, 
            prob=1,
            rotate_range=rotate_range,
            scale_range=scale_range,
            translate_range=translate_range,
            shear_range=shear_range,
            padding_mode='zeros'
        )
        
        return augment(data)

class GaussianBlurAugment_V2(MonaiAugmentationExtender):
    def __init__(self, modality_p, keys=['image'], *args, **kwargs):
        super().__init__(*args, keys=keys, **kwargs)
        self.modality_p = modality_p
        self.base_augmentation = RandGaussianSmooth(
            prob=1,
            sigma_x=(0.5,1.5),
            sigma_y=(0.5,1.5),
            sigma_z=(0.5,1.5),
        )
        # print(self.p)
        # print(self.modality_p)

    def __call__(self, data):
        if np.random.uniform(0,1) > self.p:
            # print("returning")
            return data

        for key in self.keys:
            key_data = data[key].clone()
            for channel in range(key_data.shape[0]):
                if np.random.uniform(0,1) < self.modality_p:
                    channel_data = key_data[channel].unsqueeze(0)
                    channel_data = self.base_augmentation(channel_data)
                    key_data[channel] = channel_data

            data[key] = key_data
        return data


class MonaiPairedPadToShape2d_V2:
    def __init__(self, target_shape, padding_mode="constant", padding_value=0, keys=['image', 'mask', 'label']):
        self.target_shape = target_shape
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.keys = keys

    def __call__(self, data):
        for key in self.keys():
            img = data[key]
        
            _, h, w = img.shape
            target_h, target_w = self.target_shape
    
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
    
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            img = F.pad(img, padding, mode=self.padding_mode, value=self.padding_value)
    
            data[key] = img
        
        return data

class MonaiCropAndPadToShape3d_V2:
    def __init__(self, target_shape, padding_mode="constant", padding_value=0, keys=['image', 'mask', 'label']):
        self.target_shape = target_shape
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            
            _, d, h, w = img.shape
            target_d, target_h, target_w = self.target_shape
    
            crop_d = max(0, d - target_d)
            crop_h = max(0, h - target_h)
            crop_w = max(0, w - target_w)
            
            pad_d = max(0, target_d - d)
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
    
            cropping = (crop_d // 2, crop_d - crop_d // 2, crop_h // 2, crop_h - crop_h // 2, crop_w, crop_w - crop_w // 2)
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2)
            
            img = img[:,cropping[0]:d-cropping[1],cropping[2]:h-cropping[3], cropping[4]:w-cropping[5]]
            
            img = F.pad(img, padding, mode=self.padding_mode, value=self.padding_value)
    
            data[key] = img
        
        return data

class GlobalZscore:
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, data):
        for key in self.keys:
            if data[key].abs().max() > 3:
                mask = data['brainmask']
                img = data[key]
                img_region = img[mask==1]
                img[mask==1] = img_region - img_region.mean() / img_region.std()
                data[key] = img
        return data

class GlobalMinMax:
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, data):
        for key in self.keys:
            if data[key].abs().max() > 5:
                img = data[key]
                data[key] = ((img - img.min()) / (img.max() - img.min())) * 8 - 3
        return data

class GlobalClamp:
    def __init__(self, vmin=-5, vmax=10, keys=['image']):
        self.vmin = vmin
        self.vmax = vmax
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key].clamp(self.vmin, self.vmax)
        return data

class SetImageDtype:
    def __init__(self, keys, dtypes):
        self.keys = keys
        self.dtypes = dtypes

    def __call__(self, data):
        for i, key in enumerate(self.keys):
            data[key] = data[key].type(self.dtypes[i])

        return data

def format_transform(data):
    return (cat([data['FLAIR'], data['brainmask']], dim=0), data['label'])


def aggressive_augmentations(image_keys=['FLAIR'], label_keys=['label', 'brainmask'], out_spatial_dims=(80, 192, 160), synthetic_realistic=True, axial_rot=False, bias_field='perlin', global_minmax=False, global_zscore=False, add_synthetic=True):
    combined_keys = image_keys + label_keys
    resizer = MonaiCropAndPadToShape3d_V2(out_spatial_dims, keys=combined_keys)
    dims = 3

    if synthetic_realistic:
        synthetic_aug = SyntheticGMMAugmentation(mean_z_temperature_cap=3.5, std_z_temperature_cap=1.5, std_weighting=0.5, keys=image_keys, wmh_mask='label')
    else:
        synthetic_aug = SyntheticGMMAugmentation(mean_z_temperature_cap=10, std_z_temperature_cap=10, std_weighting=0.5, keys=image_keys, wmh_mask='label')

    if add_synthetic:
        synthetic_aug = [synthetic_aug]
    else:
        synthetic_aug = []
        print("synthetic aug is switched off for ablation testing")

    return Compose(synthetic_aug + [
        resizer,
        AffineAugment_V2(p=0.4, spatial_size=out_spatial_dims, dims=dims, axial_only=axial_rot, allow_translate=True, allow_shear=False, keys=combined_keys, mode=['bilinear' for _ in range(len(image_keys))] + ['nearest' for _ in range(len(label_keys))]),
        GaussianNoiseAugment(p=0.15, dims=dims, keys=image_keys),
        GaussianBlurAugment_V2(p=0.15, modality_p=0.5, dims=dims, keys=image_keys),
        ArtefactsAugmentation(bias_field='perlin'),
        GammaAugmentation(p=0.15, allow_invert=True, dims=dims, keys=image_keys),
        MirrorAugment(p=0.5, dims=dims, keys=combined_keys)
    ] + ([GlobalMinMax(image_keys)] if global_minmax else []) + ([GlobalZscore(image_keys)] if global_zscore else []) + [
        GlobalClamp(keys=image_keys),
        SetImageDtype(keys=combined_keys, dtypes=[float32 for _ in range(len(image_keys))] + [long for _ in range(len(label_keys))]),
        format_transform
    ])

def get_val_transforms(image_keys=['FLAIR'], label_keys=['label', 'brainmask'], out_spatial_dims=(80, 192, 160), use_format_transform=True):
    combined_keys = image_keys + label_keys
    resizer = MonaiCropAndPadToShape3d_V2(out_spatial_dims, keys=combined_keys)
    dims = 3

    return Compose([
        resizer,
        SetImageDtype(keys=combined_keys, dtypes=[float32 for _ in range(len(image_keys))] + [long for _ in range(len(label_keys))]),
        
    ] + ([format_transform] if use_format_transform else []))


