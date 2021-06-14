"""
Made by Michal Borsky, 2019, copyright (C) RU
Decode a Deep Neural Network.
"""
import os
from os import path
import torch
import sleepat
from sleepat import io, nnet, utils, steps, opts


def decode_nnet(data_dir:str, lang_dir:str, exp_dir:str, config_mdl:str=None, config_ds:str=None,
    config_feats:str=None) -> None:
    """
    Decode with a DNN model saved in exp_dir. The dataset is a SimpleDataset object.
    See SimpleDataset() for content of  config_dataset. See DnnSnore for content of
    config_mdl. See train_mdl() for content of config_optim and  No kwargs transfer
    allowed from func. header to any config_* to avoid confusion.

    Arguments:
        data_dir .... directory with training data
        lang_dir ... directory that contains events.txt file
        exp_dir ... output directory of trainings

        <config_dataset> .... configuration file for SimpleDataset()
        <config_mdl> ... configuration file for DnnSnore()
        <config_feats> ... configuration file for targets_to_annot()
    """
    # Loads and Checks
    print(f'Decoding {data_dir} into {exp_dir}.')

    decode_dir = path.join(exp_dir,f'decode_{path.basename(data_dir)}')
    device_fid = path.join(exp_dir,'device')
    events = io.read_scp(path.join(lang_dir,'events'))

    for item in [data_dir,lang_dir]:
        if not path.isdir(item):
            print(f'Error: {item} does not exist.')
            exit(1)
    if not path.isdir(decode_dir):
        os.mkdir(decode_dir)
    if not path.exists(device_fid):
        print(f'Error: {device_fid} does not exist.')
        exit(1)
    if not events:
        print(f'Error: events is empty.')
        exit(1)

    conf = opts.TrainNnet(config_mdl, config_ds)
    device = io.read_scp(device_fid)['device']        
    events_num = len(set(events.values()))

    # Create dataset object
    dataset = getattr(nnet,conf.dataset)
    evalds = dataset(data_dir,config=config_ds, mode='eval')
    evalds.init_infer(events)
    sample_dim = evalds.sample_dim

    # Init a blank model and load from save
    final = path.join(exp_dir,'final.pth')
    if not path.exists(final):
        print(f'Error: {final} does not exist.')
        exit(1)
    model = getattr(nnet,conf.model)
    mdl = model(config=config_mdl, in_dim=sample_dim, out_dim=events_num)
    mdl.load_state_dict(torch.load(final)['state_dict'])
    mdl.to(device)
    mdl.eval()

    # Inference
    for batch in evalds.to_dataloader():
        sample = batch['sample'].to(device)
        output = mdl(sample)
        post = torch.nn.functional.softmax(output,dim=1)
        evalds.insert_post(post, batch['index'])

    # Make post files which contain frame-level posteriors
    (post_scp, trans_scp) = dict(),dict()
    post_dir = path.join(decode_dir,'post')
    if not path.isdir(post_dir):
        os.mkdir(post_dir)
    else:
        for file in utils.list_files(post_dir):
            os.remove(path.join(post_dir,file))

    for (utt_id,tstamp,post) in evalds.return_data('post'):
        pred = post.argmax(axis=1)
        trans = utils.targets_to_scoring(pred,post,events,config_feats,tstamp=tstamp)
        trans_scp[utt_id] = trans

        # Create posterior files
        post_fid = path.join(post_dir,f'{utt_id}.post.npy')
        io.write_npy(post_fid,post)
        post_scp[utt_id] = post_fid

    io.write_scp(path.join(decode_dir,'post.scp'),post_scp)
    io.write_scp(path.join(decode_dir,'trans'),trans_scp)
    steps.score_detect(data_dir, lang_dir, decode_dir)

    print(f'Decoding done.')

