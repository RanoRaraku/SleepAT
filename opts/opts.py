"""
    Made by Michal Borsky, 2019, copyright (C) RU
    Library with default values for all functions.
"""
from inspect import getargspec
import sleepat.io as io

## Base Options Class
## All other classes adopt functionality but define their own attributes
#-----------------------------------------------------------------
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

    def update(self, config:str=None, **kwargs):
        if config is not None:
            self.update_from_config(config)
        if kwargs:
            self.update_from_kwargs(kwargs)

    def as_kwargs(self):
        return self.__dict__

## Python Opts Class
## To get args from a Python-native class/function
#-----------------------------------------------------------------
class PyClass(BaseOpts):
    def __init__(self, pyclass, config:str=None, **kwargs):
        spec = getargspec(pyclass)
        for i,val in enumerate(reversed(spec.defaults)):
            setattr(self,spec.args[-(i+1)],val)
        self.update(config,**kwargs)

## DSP Options
#-----------------------------------------------------------------
class TimeToFrame(BaseOpts):
    def __init__(self):
        self.wstep = 0.01
        self.wlen = 0.025
        self.fs = 8000.0

class Segment(BaseOpts):
    def __init__(self):
        self.register_from_opts(TimeToFrame())
        self.remove_dc = True
        self.wtype = 'hamming'

class Spect(BaseOpts):
    def __init__(self):
        self.register_from_opts(Segment())

class PowSpect(BaseOpts):
    def __init__(self):
        self.register_from_opts(Segment())

class Preemph(BaseOpts):
    def __init__(self):
        self.preemphasis_alpha=0.97

class Mel(BaseOpts):
    def __init__(self):
        self.f = 0.0

class InvMel(BaseOpts):
    def __init__(self):
        self.melf = 0.0

class Bark(BaseOpts):
    def __init__(self):
        self.f = 0.0

class InvBark(BaseOpts):
    def __init__(self):
        self.barkf = 0.0

class Melfb(BaseOpts):
    def __init__(self):
        self.fs = 8000.0
        self.mel_filts = 22
        self.fmin = 0.0
        self.fmax = self.fs/2
        self.nfft = 200

class Barkfb(BaseOpts):
    def __init__(self):
        self.fs = 8000.0
        self.fmin = 0.0
        self.fmax = self.fs/2
        self.nfft = 200

class Delta(BaseOpts):
    def __init__(self):
        self.delta_window = 2


## Feat Options
#-----------------------------------------------------------------
class Fbank(BaseOpts):
    """
    This is questionable as it loads options from Melfb().
    Maybe throw away whole module and include apply_dct option
    in respective methods.
    """
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'FbankOpts'
        self.register_from_opts(Preemph())
        self.register_from_opts(PowSpect())
        self.register_from_opts(Melfb())
        self.fbank_type = 'melfb'
        self.use_log_fbank = True
        self.update(config,**kwargs)

class Mfcc(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'MfccOpts'
        self.register_from_opts(Preemph())
        self.register_from_opts(PowSpect())
        self.register_from_opts(Melfb())
        self.nceps = 13
        self.use_log_fbank = True
        self.update(config,**kwargs)

class Bfcc(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'BfccOpts'
        self.register_from_opts(Preemph())
        self.register_from_opts(PowSpect())
        self.register_from_opts(Barkfb())
        self.nceps = 13
        self.use_log_fbank = True
        self.update(config,**kwargs)

class SpectSlope(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'SpectSlopeOpts'
        self.register_from_opts(PowSpect())
        self.ss_bands = [0, self.fs/2]
        self.update(config,**kwargs)

class C1(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'C1'
        self.register_from_opts(Segment())
        self.update(config,**kwargs)

class Acf(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'Acf'
        self.register_from_opts(Segment())
        self.register_from_opts(Preemph())        
        self.actype = 'same'
        self.nacf = 320
        self.update(config,**kwargs)

class AddDelta(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'AddDelta'
        self.register_from_opts(Delta())
        self.delta_order = 2
        self.update(config,**kwargs)

class SpliceFrames(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'SpliceFramesOpts'
        self.splice_left = 2
        self.splice_right = 2
        self.splice_mode = 'edge'
        self.update(config,**kwargs)

class ApplyLifter(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'SpliceLifterOpts'
        self.lifter_order = 22
        self.update(config,**kwargs)

class ApplyMvn(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'ApplyMvnOpts'
        self.norm_vars = False
        self.update(config,**kwargs)

class ApplyMa(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'ApplyMaOpts'
        self.ma_weights = [.33,.33,.33]
        self.ma_mode = 'two-way'
        self.update(config,**kwargs)

class ComputePca(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'ComputePcaOpts'
        self.splice_frames = True
        self.apply_ma = False
        self.apply_mvn = True
        self.apply_mvs = False
        self.add_delta = True
        self.update(config,**kwargs)

        if self.apply_ma:
            self.register_from_opts(ApplyMa())
        if self.splice_frames:
            self.register_from_opts(SpliceFrames())
        if self.apply_mvn or self.apply_mvs:
            self.register_from_opts(ApplyMvn())
        if self.add_delta:
            self.register_from_opts(AddDelta())
        self.update(config,**kwargs)


        self.update(config,**kwargs)

class ApplyPca(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'ApplyPcaOpts'
        self.pca_dim = 40
        self.update(config,**kwargs)

## Utils Options
#-----------------------------------------------------------------
class AnnotToTargets(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'AnnotToTargetsOpts'
        self.register_from_opts(TimeToFrame())
        self.update(config,**kwargs)

class TargetsToAnnot(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'TargetsToAnnotOpts'
        self.register_from_opts(TimeToFrame())
        self.register_from_opts(TimeStamp())
        self.update(config,**kwargs)

class SegmentData(BaseOpts):
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'CreateSegmentsOpts'
        self.seg_len = 10.0
        self.omit_wave = True
        self.update(config,**kwargs)

## egs/locals
#-----------------------------------------------------------------
class PrepVSN_10048(BaseOpts):
    """
    Default options for prep_vsn10_048 method
    used to process VSN-10-048 dataset.
    """
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'PrepVSN_10048'
        self.scorings = ['ms_snore_v1_p1','ms_snore_v1_p2']
        self.bad_spk = ['']
        self.use_period = 'analysis'
        self.utt2seg = {}
        self.update(config,**kwargs)

class FormatVSN_10048(BaseOpts):
    """
    Default options for format_vsn10_048 method
    used to process VSN-10-048 dataset.
    """
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'FormatVSN_10048'
        self.valid_events = ['snorebreath','breathing-effort']
        self.channel = 'Audio'
        self.update(config,**kwargs)

## Nnet Options
#-----------------------------------------------------------------
class TrainTorch(BaseOpts):
    def __init__(self,config:str=None, **kwargs):
        self.__name__ = 'TrainTorchOpts'
        self.max_epochs = 100
        self.save_epochs = [1,5,10,15,20,30,40,50,60,70,80,90,100]
        self.optim ='SGD'
        self.loss = 'CrossEntropyLoss'
        self.update(config,**kwargs)

class Cnn_1c2h(BaseOpts):
    def __init__(self,config:str=None, **kwargs):
        self.__name__='Cnn_1c2h'
        self.in_dim = 39
        self.out_dim = 2
        self.hid_dim = 512
        self.dropout_prob = 0.5
        self.filts = 32
        self.update(config,**kwargs)

class Dnn_4h1bn(BaseOpts):
    def __init__(self,config:str=None, **kwargs):
        self.__name__='Dnn_4h1bn'
        self.in_dim = 39
        self.out_dim = 2
        self.hid_dim = 512
        self.dropout_prob = 0.5
        self.bottleneck_dim = 40
        self.update(config,**kwargs)

class Lstm_2h(BaseOpts):
    def __init__(self,config:str=None, **kwargs):
        self.__name__='Dnn_4h1bn'
        self.in_dim = 39
        self.out_dim = 2
        self.hid_dim = 128
        self.dropout_prob = 0.5
        self.seq_len = 10
        self.update(config,**kwargs)

class Dnn_2h(BaseOpts):
    def __init__(self,config:str=None, **kwargs):
        self.__name__='Dnn_2h'
        self.in_dim = 39
        self.out_dim = 2
        self.hid_dim = 512
        self.dropout_prob = 0.5
        self.update(config,**kwargs)

class SimpleDataset(BaseOpts):
    """
    Default options for SimpleDataset class used to train a
    DNN model. The DataLoader options are not here.
    """
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'SimpleDatasetOpts'
        self.splice_frames = True
        self.apply_ma = False
        self.apply_mvn = True
        self.apply_mvs = False
        self.add_delta = True
        self.mode = 'infer'
        self.update(config,**kwargs)

        if self.apply_ma:
            self.register_from_opts(ApplyMa())
        if self.splice_frames:
            self.register_from_opts(SpliceFrames())
        if self.apply_mvn or self.apply_mvs:
            self.register_from_opts(ApplyMvn())
        if self.add_delta:
            self.register_from_opts(AddDelta())
        self.update(config,**kwargs)

class ConvDataset(BaseOpts):
    """
    Default options for ConvDataset class used to train a
    CNN model. The DataLoader options are not here.
    """
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'ConvDatasetOpts'
        self.context_left = 4
        self.context_right = 4
        self.apply_mvn = True
        self.apply_mvs = False
        self.add_delta = True
        self.mode = 'infer'
        self.update(config,**kwargs)

        if self.apply_mvn or self.apply_mvs:
            self.register_from_opts(ApplyMvn())
        if self.add_delta:
            self.register_from_opts(AddDelta())
        self.update(config,**kwargs)

class SeqDataset(BaseOpts):
    """
    Default options for SeqDataset class used to train a
    DNN model. The DataLoader options are not here.
    """
    def __init__(self, config:str=None, **kwargs):
        self.__name__ = 'SeqDatasetOpts'
        self.seq_len = 100
        self.splice_frames = True
        self.apply_ma = False
        self.apply_mvn = True
        self.apply_mvs = False
        self.add_delta = True
        self.apply_pca = True
        self.mode = 'infer'
        self.update(config,**kwargs)

        if self.apply_ma:
            self.register_from_opts(ApplyMa())
        if self.splice_frames:
            self.register_from_opts(SpliceFrames())
        if self.apply_mvn or self.apply_mvs:
            self.register_from_opts(ApplyMvn())
        if self.add_delta:
            self.register_from_opts(AddDelta())
        if self.apply_pca:
            self.register_from_opts(ApplyPca())
        self.update(config,**kwargs)

## Steps Options
#----------------------------------------------------------------
class TrainNnet(BaseOpts):
    def __init__(self,config_mdl:str=None, config_ds:str=None, **kwargs):
        self.__name__='NnetOpts'
        self.model = 'Dnn_4h1bn'
        self.dataset = 'SimpleDataset'
        self.update(config_mdl,**kwargs)
        self.update(config_ds,**kwargs)

## Objects Options
#--------------------------------------------------------------
class TimeStamp(BaseOpts):
    def __init__(self,config:str=None, **kwargs):
        self.__name__='TimeStamp'
        self.format = '%Y/%m/%dT%H:%M:%S.%f'
        self.stamp = '0000/01/01T00:00:000000'
        self.offset = 0.0
        self.update(config,**kwargs)