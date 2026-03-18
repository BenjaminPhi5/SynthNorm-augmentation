from typing import Dict
import synthnormaug
import numpy as np
import importlib.resources as pkg_resources
import os
import torch

WEIGHTS_NAMES = {
    "FLAIR":"flair_gmm_params.npz",
     "T1w":"t1_gmm_params.npz",
     "T2w":"t2_gmm_params_multi.npz",
}

def load_gm_data(imgtype='FLAIR', gmm_weight_path='.'):
    if imgtype not in WEIGHTS_NAMES:
        raise ValueError(f"no gmm weight config file for imgtype: {imgtype}, must be one of {WEIGHTS_NAMES.keys()}")
    
    weights_path = os.path.join(
        gmm_weight_path,
        WEIGHTS_NAMES[imgtype]
    )
    return np.load(weights_path)
        

synthseg_keys = {
    'Cerebral White Matter':[2, 41],
    'Cerebral Cortex':[3, 42],
    'Lateral Ventricles':[4, 43],
    'Inferior Lateral Ventricles':[5, 44],
    'Cerbellum White Matter':[7, 46],
    'Cerebellum Cortex':[8, 47],
    'Thalamus':[10, 49],
    'Caudate':[11, 50],
    'Putamen':[12, 51],
    'Pallidum':[13, 52],
    '3rd Ventricle':[14],
    '4th Ventricle':[15],
    'Brain Stem':[16],
    'Hippocampus':[17, 53],
    'Amygdala':[18, 54],
    'CSF':[24],
    'Accumbens':[26, 58],
    'Ventral Diencephalon':[28, 60]
}

class GaussianMixtureRenormalizer():
    def __init__(self, synthseg_keys:Dict[str, int], gm_data:Dict, mean_z_temperature_cap:float=3.5, std_z_temperature_cap:float=1.5, min_std:float=0.05, std_weighting=0.7, mean_only:bool=False, csf_xscale:float=0.5, csf_xshift:float=4, apply_csf_correction=True):
        self.synthseg_keys = synthseg_keys
        self.key_indexes = {key:i for i,key in enumerate(synthseg_keys.keys())}
        
        self.weights = gm_data['weights']
        self.Nc = gm_data['Nc']
        self.means = gm_data['means']
        self.covariances_cholesky = gm_data['covariances_cholesky']
        self.mean_z_temperature_cap = mean_z_temperature_cap
        self.std_z_temperature_cap = std_z_temperature_cap
        self.mean_only = mean_only
        self.min_std = min_std
        self.std_weighting = std_weighting
        
        self.num_zscore_regions = self.means[0].shape[0] // 2

        self.csf_xscale = csf_xscale
        self.csf_xshift = csf_xshift
        self.apply_csf_correction = apply_csf_correction

    def sample_zscore_params(self):
        ### determine the noise vector z used to generate the sample
        # with separate temperatures for the 
        mean_z_temperature = np.random.rand() * self.mean_z_temperature_cap
        std_z_temperature = np.random.rand() * self.std_z_temperature_cap

        z_mean = np.random.normal(loc=0, scale=mean_z_temperature, size=self.num_zscore_regions)
        z_std = np.random.normal(loc=0, scale=std_z_temperature, size=self.num_zscore_regions)
        z = np.concatenate([z_mean, z_std])

        ### select the component from the gaussian mixture
        m = np.random.choice(self.Nc, p=self.weights)
        mu = self.means[m]
        A = self.covariances_cholesky[m]
        
        ### generate the zscore params for each ROI
        # force the std parameters to be greater than 0
        s = z @ A + mu
        s[self.num_zscore_regions:] = np.maximum(s[self.num_zscore_regions:], self.min_std) 
        s_mean = s[:self.num_zscore_regions]
        s_std = s[self.num_zscore_regions:]
        
        return s_mean, s_std

    def scaled_sigmoid(self, x, xscale=0.5, xshift=4):
        return 1/(1+ torch.exp(-(x/xscale - xshift)))

    def renormalize_image(self, img, synthseg, wmh=None):
        img = img.clone()

        if wmh is not None: # correct other anatomical regions (such as ventricles, basal ganglia) in synthseg map overlapping with WMH
            synthseg[wmh==1] = self.synthseg_keys['Cerebral White Matter'][0] # hemispheres are given the same norm parameters so this is fine
        
        s_mean, s_std = self.sample_zscore_params()

        for key, region_ids in synthseg_keys.items():
            key_i = self.key_indexes[key]
            if len(region_ids) == 2:
                region = (synthseg == region_ids[0]) | (synthseg == region_ids[1])
            else:
                region = (synthseg == region_ids[0])
    
            img_region = img[region]
            region_mean = img_region.mean()
            region_std = img_region.std()

            # take the weighted average for CSF regions due to poor alignment of the synthseg causing drastic
            # differences in image intensity
            # weighted average is based on number of standard deviations from the mean
            if key in ['CSF', 'Inferior Lateral Ventricles', 'Lateral Ventricles', '3rd Ventricle', '4th Ventricle'] and self.apply_csf_correction: # take weighted average of new parameters and original z-score parameters
                stds_from_mean = (img_region - region_mean).abs() / region_std
                scale = self.scaled_sigmoid(stds_from_mean, self.csf_xscale, self.csf_xshift)
                new_mean = s_mean[key_i] * (1 - scale) + (scale) * region_mean
                new_std = s_std[key_i] * (1 - scale) + (scale) * region_std

            else:
                new_mean = s_mean[key_i]
                new_std = s_std[key_i] * self.std_weighting + region_std * (1 - self.std_weighting)
            
            if self.mean_only:
                img[region] = img[region] - region_mean + new_mean
            else:
                img[region] = ((img[region] - region_mean) / region_std) * new_std + new_mean
    
        return img
        
class SyntheticGMMAugmentation:
    def __init__(self, 
                 keys=['FLAIR', 'T1w'],
                 mean_z_temperature_cap:float=3.5,
                 std_z_temperature_cap:float=1.5,
                 min_std:float=0.05,
                 std_weighting=0.7,
                 mean_only:bool=False,
                 csf_xscale:float=0.5,
                 csf_xshift:float=4,
                 wmh_mask=None,
                 drop_synthseg=True,
                 apply_csf_correction=True,
                 gmm_weight_path='.',
                ):

        """
        SynthNorm Gaussian Mixture Model (GMM)-based intensity augmentation for MRI.
    
        This class implements the core SynthNorm augmentation strategy, where
        region-wise intensity statistics (mean and standard deviation) are
        resampled from a pretrained Gaussian Mixture Model (GMM) and applied
        to an input MRI volume using anatomical volumes provided from SynthSeg.
    
        The augmentation preserves intra-region texture while modifying global
        tissue intensity characteristics to simulate variations across scanners and
        acquisition protocols.
    
        Parameters
        ----------
        keys : list of str, optional
            Image keys in the input dictionary to which augmentation will be applied
            (e.g. ['FLAIR', 'T1w']) or just ['FLAIR'].
    
        mean_z_temperature_cap : float, optional
            Upper bound on z-score scaling applied to sampled means from the GMM.
            Controls how extreme the sampled intensity shifts can be.
    
        std_z_temperature_cap : float, optional
            Upper bound on z-score scaling applied to sampled standard deviations.
    
        min_std : float, optional
            Minimum allowable standard deviation for any region. Prevents erasure
            of tissue texture (i.e. std = 0).
    
        std_weighting : float, optional
            Interpolation weight between original and sampled standard deviation
            for non-CSF tissues. Helps preserve pathological structures (e.g. WMH).
    
        mean_only : bool, optional
            If True, only region means are modified while standard deviations are
            kept unchanged.
    
        csf_xscale : float, optional
            Scaling factor used in sigmoid-based interpolation for CSF regions.
            Controls how aggressively sampled parameters are blended with original
            statistics near tissue boundaries.
    
        csf_xshift : float, optional
            Shift parameter for sigmoid interpolation in CSF regions.
    
        wmh_mask : str or None, optional
            WMH (white matter hyperintensity) mask name in the input dictionary.
            If provided, WMH voxels are reassigned to white matter
            prior to computing region statistics.
    
        drop_synthseg : bool, optional
            If True, removes the 'synthseg' segmentation from the output dictionary
            after augmentation.
    
        apply_csf_correction : bool, optional
            Whether to apply voxel-wise interpolation for CSF and ventricular regions
            to reduce boundary artefacts.
    
        gmm_weight_path : str, optional
            Path to the directory (folder path) containing pretrained GMM parameter files (.npy).
    
        Notes
        -----
        - Requires a SynthSeg-style segmentation map under key 'synthseg' when calling the function.
        - Augmentation is applied independently per modality if both FLAIR and T1w are provided..
        """

        self.keys = keys
        self.drop_synthseg = drop_synthseg

        gm_transformers = {}
        for key in keys:
            gm_data = load_gm_data(key, gmm_weight_path=gmm_weight_path)
            gm_transformers[key] = GaussianMixtureRenormalizer(synthseg_keys, gm_data, mean_z_temperature_cap=std_z_temperature_cap, std_z_temperature_cap=std_z_temperature_cap, min_std=min_std, std_weighting=std_weighting, mean_only=mean_only, csf_xscale=csf_xscale, csf_xshift=csf_xshift, apply_csf_correction=apply_csf_correction)

        self.gm_transformers = gm_transformers
        self.wmh_mask=wmh_mask

    def __call__(self, data):
        data = {key:value for key,value in data.items()}
        synthseg = data['synthseg']
        wmh=None
        if self.wmh_mask:
            wmh = data[self.wmh_mask]

        for key in self.keys:
            data[key] = self.gm_transformers[key].renormalize_image(data[key], synthseg, wmh)

        if self.drop_synthseg:
            del data['synthseg']
        
        return data 

