"""
Made by Michal Borsky, 2019, copyright (C) RU

Transforms prepared data into a PyTorch-native dataset. A dataset
consists of features, targets, spk2utt, utt2spk, mvn.scp, and
optionally configuration for feature processing. Feature processing
happens during item fetching. Every Dataset class contains
.to_dataloader() routine to convert in to PyTorch.*.DataLoader
object for further functionality.
"""
import os
from os import path
import numpy as np
import torch
from torch.utils import data
import sleepat
from sleepat import io, opts, feat, utils, opts

class SimpleDataset(data.Dataset):
    """
    Simple dataset for PyTorch. The input is a data directory that contains
    feats.scp,targets.scp,spk2utt,utt2spk. All data is loaded to memory, so
    not suitable for very large datasets. 

    The output is a dictionary with "sample" and "target" keys. The sample is
    a tensor (BS,FD)-dimension, where BS is batch size, and FD is feature
    dimension. Target has (BS,) dimensionality. The class is suitable for a 
    flat DNN architecture. The dataset will likely be used with DataLoader,
    which adds batching and other functionality.
    """

    def __init__(self, data_dir:str, config:str=None, **kwargs):
        """
        Optional args <> can be set from a config file or as **kwargs.
        DataLoader args can be specified as well but are loaded
        dynamically, check your Torch version for defaults.

        Arguments:
        data_dir ... directory with feats.scp,targets.scp,spk2utt,utt2spk
        <apply_lda> ... whether to apply LDA (def:bool = True)
            <lda_dim> ... LDA output dimensionality (def:int = 40)
        <splice_frames> ... whether to apply frame splicing (def:bool = True)
            <splice_left> ... no. frames to left to splice onto center frame (def:int = 2)
            <splice_right> ... no. frames to right to splice onto center frame (def:int = 2)
            <splice_mode> ... how to treat edge values, default is to repeat them (def:str='edge')
        <add_delta> ... whether to add dynamic coefficients (def:bool = True)
            <delta_window> ... no +-context frames to delta computation (def:int = 2)
            <delta_order> ... order of delta coefficients (def:int = 2)
        <apply_mvs> ... whether to apply mean and variance subtraction (def: bool=False)
            <norm_vars> ... whether to normaliza variances (def: bool=False)
        <apply_mvn> ... whethet to apply mean and variance normalization (def: bool=False)
            <norm_vars> ... whether to normaliza variances (def: bool=False)
        <<DataLoader args>> ... see torch.utils.data.DataLoader for args.
        config ... a configuration JSON file to specify all above
        **kwargs ... setting all above through kwargs
        """
        self.conf = opts.SimpleDatasetOpts(config,**kwargs)
        self.conf_dl = opts.PyClassOpts(data.DataLoader,config,**kwargs)
        self.feats_scp = io.read_scp(path.join(data_dir,'feats.scp'))
        self.targets_scp = io.read_scp(path.join(data_dir,'targets.scp'))
        self.mvn_scp = io.read_scp(path.join(data_dir,'mvn.scp'))
        self.spk2utt = io.read_scp(path.join(data_dir,'spk2utt'))
        self.utt2spk = io.read_scp(path.join(data_dir,'utt2spk'))
        self.annot = io.read_scp(path.join(data_dir,'annotation'))
        if os.path.exists(path.join(data_dir,'periods')):
            self.periods = io.read_scp(path.join(data_dir,'periods'))

        utils.validate_data(data_dir, no_feats=False)
        self.set_sample_size()
        self.set_sample_dim()
        self.set_target_dim()
        self.make_samples()

    def __len__(self):
        """
        Return sample size.
        """
        return self.sample_size

    def __getitem__(self, idx:int):
        """
        Overwrites default method from torch.utils.data.Dataset class.
        Uses as_tensor() so data is not copied, might be safer to use tensor().
        """
        return {"target": torch.as_tensor(self.targets[idx]),
            "sample": torch.as_tensor(self.samples[idx]),
            "index": idx}

    def set_sample_size(self):
        """
        Set total number of samples and targets for indexing. Corresponds
        to number of accessible feature vectors, which can shrink due to 
        how the edge vectors are treated during feature processing.
        """
        if len(self.feats_scp) == 0:
            print('SimpleDataset.set_sample_size(): No samples in your dataset, (empty feats.scp?)')
            exit(1)

        sample_num = 0
        for (_,m) in utils.feats_to_len(self.targets_scp):
            if self.conf.splice_frames and self.conf.splice_mode == 'empty':
                m -= (self.conf.splice_left + self.conf.splice_right)
            sample_num += m

        if sample_num == 0:
            print('SimpleDataset.make_meta(): No samples in your dataset, (empty feats.scp?)')
            exit(1)
        self.sample_size = sample_num

    def set_target_dim(self):
        """
        Set output target dimension. Generally the targets are scalars
        so target_dim = 0, knows nothing about how many targets. We assume
        all targets have the same dimensionality.
        """
        if len(self.targets_scp) == 0:
            print('SimpleDataset.set_target_dim(): No targets for your dataset, (empty targets.scp?)')
            exit(1)
        (_,n) = next(utils.feats_to_dim(self.targets_scp))
        self.target_dim = n

    def set_sample_dim(self):
        """
        Set output sample dimension, takes into account feature post-processing.
        Assumes the features are 2D so sample_dim is the dimension of a vector,
        knows nothing about how many vectors. We assume all samples have the same
        dimensionality.
        """
        if len(self.feats_scp) == 0:
            print('SimpleDataset.set_sample_dim(): No samples in your dataset, (empty feats.scp?)')
            exit(1)

        (_,n) = next(utils.feats_to_dim(self.feats_scp))
        n = n[0]

        if self.conf.add_delta:
            n *= (1+self.conf.delta_order)
        if self.conf.splice_frames:
            n *= (1+self.conf.splice_left+self.conf.splice_right)
        if self.conf.apply_lda:
            n = self.conf.lda_dim
        self.sample_dim = (n,)

    def make_samples(self):
        """
        Create training samples. Everything is loaded into memory and
        indexed by self.meta to trace back utterance_ and frame_ids.
        Meta file is indexed by utt_id and stores indexes of an utt
        with respect to samples/target and the annotation. Indexes
        define a half-open Pythonic interval <beg,end).
        """
        sample_num = (self.sample_size,)
        self.meta = dict()
        self.targets = np.empty(shape=sample_num+self.target_dim, dtype=np.int32)
        self.samples = np.empty(shape=sample_num+self.sample_dim, dtype=np.float32)

        beg = 0
        for utt_id, spk_id in self.utt2spk.items():
            feats = io.read_npy(self.feats_scp[utt_id])
            targets = io.read_npy(self.targets_scp[utt_id])
            if feats.shape[0] !=  targets.shape[0]:
                print(f'SimpleDataset.make_samples(): No. feats does not match no. targets for {utt_id}.')
                exit(1)

            end = beg + feats.shape[0]
            feats = self.process_feats(feats, spk_id)
            self.samples[beg:end] = feats
            self.targets[beg:end] = targets
            self.meta[utt_id] = {'beg':beg, 'end':end}
            beg = end

    def to_dataloader(self):
        """
        Transform dataset into the DataLoader object. All dataloader
        arguments can be specified in a config file.
        """
        return data.DataLoader(self, **self.conf_dl.as_kwargs())

    def process_feats(self, feats:np.ndarray, spk_id:str):
        """
        Custom feature processing chain, specific for each situation.
        """
        if self.conf.apply_mvs:
            feats = feat.apply_mvn(feats,**self.conf.as_kwargs())
        if self.conf.apply_mvn:
            feats = feat.apply_mvn(feats,self.mvn_scp[spk_id],**self.conf.as_kwargs())
        if self.conf.add_delta:
            feats = feat.add_delta(feats,**self.conf.as_kwargs())
        if self.conf.splice_frames:
            feats = feat.splice_frames(feats,**self.conf.as_kwargs())
        if self.conf.apply_lda:
            feats = feats
        return feats

    def init_decode(self, class_num:dict=None):
        """
        Prepare prediction and posterior arrays that match dimensions
        and data type of targets. Optional during decoding step to help
        evaluate model, smooth prediction etc.

        Potentially dissociate creating pred and post
        """
        self.pred = np.empty_like(self.targets)
        if class_num is not None:
            self.post = np.empty(shape=(self.sample_size,class_num),dtype=np.float32)

    def insert_pred(self, pred, idx):
        """
        Insert prediction into correct place in self.pred.
        Prediction and index is either a torch tensor or a
        numpy array. Values can come as a batch or as a sample.
        """
        if torch.is_tensor(pred):
            if pred.is_cuda:
                pred = pred.cpu()
            if pred.requires_grad:
                pred = pred.detach()
            pred = pred.numpy()
        if torch.is_tensor(idx):
            if idx.is_cuda:
                idx = idx.cpu()
            idx = idx.numpy()
        self.pred[idx] = pred.astype(self.targets.dtype)

    def insert_post(self, post, idx):
        """
        Insert posteriors into correct place in self.post.
        Posteriors and index is either a torch tensor or a
        numpy array. Values can come as a batch or as a sample.
        """
        if torch.is_tensor(post):
            if post.is_cuda:
                post = post.cpu()
            if post.requires_grad:
                post = post.detach()                
            post = post.numpy()
        if torch.is_tensor(idx):
            if idx.is_cuda:
                idx = idx.cpu()
            idx = idx.numpy()
        self.post[idx] = post.astype(np.float32)

    def align_array(self, array:str='feats') -> tuple:
        """
        Align targets/feats/prediction/posteriors with utterance
        as defined in self.meta. Used prior to i.e tranforming
        pred/post to annotation or doing evaluation on per-utt basis.
        Due to feature processing, output can be shorter than original
        arrays. Output is a stream of tuples (utt_id, time_stamp, array).
        """
        vect = getattr(self, array)
        for utt_id, item in self.meta.items():
            if hasattr(self,'periods'):
                tstamp = self.periods[utt_id]['start']
            else:
                tstamp = ''
            yield(utt_id,tstamp,vect[item['beg']:item['end']])
