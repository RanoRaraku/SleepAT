"""
Made by Michal Borsky, 2019, copyright (C) RU

Transforms prepared data into a PyTorch-native dataset. A dataset consists of 
features, targets, spk2utt, utt2spk, mvn.scp, and optionally configuration
for feature processing. Feature processing happens during item fetching.
Every Dataset class contains .to_dataloader() routine to convert in to
PyTorch.*.DataLoader object for further functionality.
"""
from os.path import join
import numpy as np
from torch import tensor, as_tensor     #pylint disable=no-name-in-module
from torch.utils.data import Dataset, DataLoader
from sleepat.utils import feats_to_len, feats_to_dim
from sleepat.base.opts import ConvDatasetOpts, PyClassOpts
from sleepat.io import read_scp, read_npy
from sleepat.feat import add_delta, apply_mvn

class ConvDataset(Dataset):
    """
    Convolution dataset for PyTorch. The input is a data directory that contains
    feats.scp,targets.scp,spk2utt,utt2spk. All data is loaded to memory, so
    not suitable for very large datasets.

    The output is a dictionary with "sample" and "target" keys. The sample is
    a tensor (BS,CS,FD)-dimension, where BS is batch size, CS is context size,
    and FD is feature dimension. Target has (BS,) dimensionality. The class is
    suitable for a CNN architecture with 2D convolution layer. The dataset will
    likely be used with DataLoader, which adds batching and other functionality.
    """
    def __init__(self, data_dir:str, config:str=None, **kwargs):
        """
        Optional args <> can be set from a config file or as **kwargs.
        DataLoader args can be specified as well but are loaded
        dynamically, check your Torch version for defaults.

        Arguments:
        data_dir ... directory with feats.scp,targets.scp,spk2utt,utt2spk
        <context_left> ... frames to left for conv. layer input (def:int = 4)
        <context_right> ... frames to right for conv. layer input (def:int = 4)
        <apply_lda> ... whether to apply LDA (def:bool = True)
            <lda_dim> ... LDA output dimensionality (def:int = 40)
        <add_delta> ... whether to add dynamic coefficients (def:bool = True)
            <delta_window> ... no +-context frames to delta computation (def:int = 2)
            <delta_order> ... order of delta coefficients (def:int = 2)
        <apply_mvs> ... whether to apply mean and variance subtraction (def: bool=False)
            <norm_vars> ... whether to normaliza variances (def: bool=False)        
        <apply_mvn> ... whethet to apply mean and variance normalization (def: bool=True)
            <norm_vars> ... whether to normaliza variances (def: bool=False)
        <<DataLoader args>> ... see torch.utils.data.DataLoader for args.
        config ... a configuration JSON file to specify all above
        **kwargs ... setting all above through kwargs
        """
        self.conf = ConvDatasetOpts(config,**kwargs)
        self.conf_DL = PyClassOpts(DataLoader,config,**kwargs)
        self.feats_scp = read_scp(join(data_dir,'feats.scp'))
        self.targets_scp = read_scp(join(data_dir,'targets.scp'))
        self.mvn_scp = read_scp(join(data_dir,'mvn.scp'))
        self.spk2utt = read_scp(join(data_dir,'spk2utt'))
        self.utt2spk = read_scp(join(data_dir,'utt2spk'))

        self.make_meta()
        self.set_sample_size()
        self.set_feats_dim()
        self.set_targets_dim()
        self.make_examples()

    def __len__(self):
        """
        Return sample size.
        """
        return self.sample_size

    def __getitem__(self, idx):
        """
        Return data and target. The output is a tensor (1,21,FD)-dimension,
        where 1 is channel, 21 is temporal context, and FD is feature dimension.
        Uses as_tensor() so data is not copied, might be safer to use tensor().
        """
        frame_idx = self.meta[idx]['sample_idx']
        sample = self.feats[frame_idx-self.conf.context_left:
            frame_idx+self.conf.context_right+1]
        return {"targets": as_tensor(self.targets[frame_idx]),
            "sample": as_tensor(sample)}

    def make_meta(self):
        """
        Create a metafile with dataset- and frame-wise indexing,
        needed for __getitem__()
        """
        self.meta = dict()

        db_idx = 0
        frames_num = 0
        for utt_id, flen in feats_to_len(self.targets_scp):
            for frame_idx in range(self.conf.context_left, flen - self.conf.context_right):
                self.meta[db_idx] = {'sample_idx': frame_idx+frames_num,'utt_frame':frame_idx, 'utt_id': utt_id}
                db_idx += 1
            frames_num += flen

        if db_idx == 0:
            print('SimpleDataset.make_meta(): No samples in your dataset, (empty feats.scp?)')
            exit(1)

    def set_sample_size(self):
        """
        Set total number of samples across all features files.
        """
        if not hasattr(self,'meta'):
            self.make_meta()
        self.sample_size = len(self.meta)

    def set_targets_dim(self):
        """
        Set targets dimension.
        """
        if len(self.targets_scp) == 0:
            print('SimpleDataset.set_targets_dim(): No targets for your dataset, (empty targets.scp?)')
            exit(1)
        self.targets_dim = feats_to_dim(self.targets_scp)

    def set_sample_dim(self):
        """
        Set output sample dimension, takes into account feature post-processing.
        We assume the features are 2D.
        """
        if len(self.feats_scp) == 0:
            print('SimpleDataset.set_sample_dim(): No samples in your dataset, (empty feats.scp?)')
            exit(1)
        m = feats_to_dim(self.feats_scp)[0]
        n = 1+self.conf.context_left+self.conf.context_right

        if self.conf.add_delta:
            m *= (1+self.conf.delta_order)
        if self.conf.apply_lda:
            m = self.conf.lda_dim
        self.sample_dim = (m,n)


    def make_examples(self):
        """
        Create training examples. Everything is loaded into memory and
        correctly indexed by self.meta.
        """
        if not hasattr(self,'meta'):
            self.make_meta()
        if not hasattr(self,'sample_size'):
            self.set_sample_size()
        if not hasattr(self,'feats_dim'):
            self.set_feats_dim()

        total = (self.sample_size + (self.conf.context_left + 
            self.conf.context_right)*len(self.utt2spk),)
        self.targets = np.empty(shape = total + self.targets_dim, dtype=np.float32)
        self.feats = np.empty(shape= total + self.feats_dim, dtype=np.float32)

        beg = 0
        for utt_id, spk_id in self.utt2spk.items():
            feats = read_npy(self.feats_scp[utt_id])
            targets = read_npy(self.targets_scp[utt_id])
            if feats.shape[0] !=  targets.shape[0]:
                print(f'SimpleDataset.make_examples(): No. feats does not match no. targets for {utt_id}.')
                exit(1)

            end = beg + feats.shape[0]
            feats = self.process_feats(feats, spk_id)
            self.feats[beg:end] = feats
            self.targets[beg:end] = targets
            beg = end

    def to_dataloader(self):
        """
        Transform dataset into the DataLoader object. All dataloader
        arguments can be speficied in config.
        """
        return DataLoader(self, **self.conf_DL.as_kwargs())

    def process_feats(self, feats, spk_id):
        """
        Custom feature processing chain, specific for each situation.
        """
        if self.conf.apply_mvs:
            feats = apply_mvn(feats, norm_vars=self.conf.norm_vars)
        if self.conf.apply_mvn:    
            feats = apply_mvn(feats, self.mvn_scp[spk_id], norm_vars=self.conf.norm_vars)
        if self.conf.add_delta:            
            feats = add_delta(feats, delta_order=self.conf.delta_order,
                delta_window=self.conf.delta_window)
        if self.conf.apply_lda:
            feats = feats
        return feats
