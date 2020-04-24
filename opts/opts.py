"""
    Made by Michal Borsky, 2019, copyright (C) RU
    Configuration Library with options for all routines
"""
from inspect import getargspec
import sleepat.io as io

## Base Options Class for functionality
## All other classes adopt functionality but define their own attributes
class BaseOpts(object):
    def __init__(self):
        self.__name__= ''

    def register_from_opts(self, opts):
        for key, item in opts.__dict__.items():
            setattr(self,key,item)

    def update_from_config(self, config:str):
        conf = io.read_scp(config)
        for key, item in conf.items():
            if hasattr(self, key):
                setattr(self, key, item)
            #else :
            #    print(f'Warning, {self.__name__} has no attribute /{key}/.')

    def update_from_kwargs(self, kwargs:dict):
        for key, item in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, item)
            #else:
            #    print(f'Warning, {self.__name__} has no attribute /{key}/.')

    def update(self, config:str = None, kwargs:dict = {}):
        if config is not None:
            self.update_from_config(config)
        if kwargs:
            self.update_from_kwargs(kwargs)

    def as_kwargs(self):
        return self.__dict__

class PyClassOpts(BaseOpts):
    def __init__(self, pyclass, config:str = None, **kwargs):
        spec = getargspec(pyclass)
        for i,val in enumerate(reversed(spec.defaults)):
            setattr(self,spec.args[-(i+1)],val)
        self.update(config,kwargs)

#-----------------------------------------------------------------
## DSP Options
class TimeToFrameOpts(BaseOpts):
    def __init__(self):
        self.wlen = 0.025
        self.wstep = 0.01

class SegmentOpts(BaseOpts):
    def __init__(self):
        self.register_from_opts(TimeToFrameOpts())
        self.fs = 8000.0
        self.remove_dc = True
        self.wtype = 'hamming'

class SpectOpts(BaseOpts):
    def __init__(self):
        self.register_from_opts(SegmentOpts())

class PowSpectOpts(BaseOpts):
    def __init__(self):
        self.register_from_opts(SegmentOpts())

class PreemphOpts(BaseOpts):
    def __init__(self):
        self.preemphasis_alpha=0.97

class MelOpts(BaseOpts):
    def __init__(self):
        self.f = 0.0

class InvMelOpts(BaseOpts):
    def __init__(self):
        self.melf = 0.0

class BarkOpts(BaseOpts):
    def __init__(self):
        self.f = 0.0

class InvBarkOpts(BaseOpts):
    def __init__(self):
        self.barkf = 0.0

class MelfbOpts(BaseOpts):
    def __init__(self):
        self.fs = 8000.0
        self.mel_filts = 22
        self.fmin = 0.0
        self.fmax = self.fs/2
        self.nfft = 200

class BarkfbOpts(BaseOpts):
    def __init__(self):
        self.fs = 8000.0
        self.fmin = 0.0
        self.fmax = self.fs/2
        self.nfft = 200

class DeltaOpts(BaseOpts):
    def __init__(self):
        self.delta_window = 2


## Feat Options
class FbankOpts(BaseOpts):
    """
    This is questionable as it loads options from MelfbOpts().
    Maybe throw away whole module and include apply_dct option
    in respective methods.
    """
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'FbankOpts'
        self.register_from_opts(PreemphOpts())
        self.register_from_opts(PowSpectOpts())
        self.register_from_opts(MelfbOpts())
        self.fbank_type = 'melfb'
        self.use_log_fbank = True
        self.update(config,kwargs)

class MfccOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'MfccOpts'
        self.register_from_opts(PreemphOpts())
        self.register_from_opts(PowSpectOpts())
        self.register_from_opts(MelfbOpts())
        self.nceps = 13
        self.use_log_fbank = True
        self.update(config,kwargs)

class BfccOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'BfccOpts'
        self.register_from_opts(PreemphOpts())
        self.register_from_opts(PowSpectOpts())
        self.register_from_opts(BarkfbOpts())
        self.nceps = 13
        self.use_log_fbank = True
        self.update(config,kwargs)

class SpectSlopeOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'SpectSlopeOpts'
        self.register_from_opts(PowSpectOpts())
        self.ss_bands = [0, self.fs/2]
        self.update(config,kwargs)

class C1Opts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'C1Opts'
        self.register_from_opts(SegmentOpts())
        self.update(config,kwargs)

class AddDeltaOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'AddDeltaOpts'
        self.register_from_opts(DeltaOpts())
        self.delta_order = 2
        self.update(config,kwargs)

class SpliceFramesOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'SpliceFramesOpts'
        self.splice_left = 2
        self.splice_right = 2
        self.splice_mode = 'edge'
        self.update(config,kwargs)

class ApplyLifterOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'SpliceLifterOpts'
        self.lifter_order = 22
        self.update(config,kwargs)

class ApplyMvnOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'ApplyMvnOpts'
        self.norm_vars = False
        self.update(config,kwargs)

class ApplyLdaOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'ApplyLdaOpts'
        self.lda_dim = 40
        self.update(config,kwargs)

## Utils Options
class AnnotToTargetsOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'AnnotToTargetsOpts'
        self.register_from_opts(TimeToFrameOpts())
        self.update(config,kwargs)

class TargetsToAnnotOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'TargetsToAnnotOpts'
        self.register_from_opts(TimeToFrameOpts())
        self.utt_timestamp = ''
        self.no_null = True
        self.update(config,kwargs)

class SegmentDataOpts(BaseOpts):
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'CreateSegmentsOpts'
        self.segm_len = 10.0
        self.segm_len_min = 1.0
        self.segm_len_max = 20.0
        self.not_wave = True
        self.update(config,kwargs)

## Dataset Options
class SimpleDatasetOpts(BaseOpts):
    """
    Default options for SimpleDataset class used to train a
    DNN model. The DataLoader options are not here.
    """
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'SimpleDatasetOpts'
        self.splice_frames = True
        self.apply_mvn = True
        self.apply_mvs = False
        self.add_delta = True
        self.apply_lda = False
        self.update(config,kwargs)

        if self.splice_frames:
            self.register_from_opts(SpliceFramesOpts())
        if self.apply_lda:
            self.register_from_opts(ApplyLdaOpts())
        if self.apply_mvn or self.apply_mvs:
            self.register_from_opts(ApplyMvnOpts())
        if self.add_delta:
            self.register_from_opts(AddDeltaOpts())
        self.update(config,kwargs)

class ConvDatasetOpts(BaseOpts):
    """
    Default options for ConvDataset class used to train a
    CNN model. The DataLoader options are not here.
    """
    def __init__(self, config:str = None, **kwargs):
        self.__name__ = 'ConvDatasetOpts'
        self.context_left = 4
        self.context_right = 4
        self.apply_mvn = True
        self.apply_mvs = False
        self.add_delta = True
        self.apply_lda = False
        self.update(config,kwargs)

        if self.apply_lda:
            self.register_from_opts(ApplyLdaOpts())
        if self.apply_mvn or self.apply_mvs:
            self.register_from_opts(ApplyMvnOpts())
        if self.add_delta:
            self.register_from_opts(AddDeltaOpts())
        self.update(config,kwargs)

## Nnet Options
class TrainMdlOpts(BaseOpts):
    def __init__(self,config:str = None, **kwargs):
        self.__name__ = 'TrainMnnOpts'
        self.max_epochs = 100
        self.save_iters = [1,5,10,15,20,30,40,50,60,70,80,90,100]
        self.update(config,kwargs)

class CnnSnoreOpts(BaseOpts):
    def __init__(self,config:str = None, **kwargs):
        self.__name__='CnnSnoreOpts'
        self.in_dim = 39
        self.out_dim = 2
        self.hid_layers = 2
        self.hid_dim = 512
        self.conv_layers = 2
        self.filts = 32
        self.update(config,kwargs)

class DnnSnoreOpts(BaseOpts):
    def __init__(self,config:str = None, **kwargs):
        self.__name__='DnnSnoreOpts'
        self.in_dim = 39
        self.out_dim = 2
        self.hid_layers = 2
        self.hid_dim = 512
        self.dropout_prob = 0.5
        self.bottleneck = True
        self.bottleneck_dim = 40
        self.update(config,kwargs)
