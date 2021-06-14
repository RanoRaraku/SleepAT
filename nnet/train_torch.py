"""
Made by Michal Borsky, 2019, copyright (C) RU.
A routine to train a Torch model.
"""
import os
from os import path
import torch
import sleepat
from sleepat import io, opts


def train_torch(mdl, trainloader, devloader, exp_dir:str, config:str=None, **kwargs) -> None:
    """
    Train a PyTorch model using the trainloader. The trainloader is PyTorch DataLoader object
    that extends a custom PyTorch DataSet object. Use nnet.SimpleDataset.to_dataloader(), i.e.,
    to create these objects. The training performs 'max_epochs' number of  epochs. Models are
    saved periodically as defined by 'save_epochs'. The best model is picked based on minimal
    loss on devloader and linked to 'final.pth'. The default optim is SGD, but any valid Torch
    optim is supported. The default loss is CrossEnropyLoss, but any valid Torch loss is supported.
    The training parameters, loss and optim can have their arguments defined by a config file
    or through kwargs. Only valid torch.loss or torch.optim args are passed to avoid crash.
    Check pyTorch docu for more info. The model is saved as a 'state_dict' so a template is needed
    to load it. We also save 'device' as dict() object into exp_dir/device in JSON format.

    Arguments:
        mdl ... torch model to train
        trainloader ... torch DataLoader class with train data
        devloader ... torch DataLoader class with development data
        exp_dir ... output directory of training
        config ... config. JSON file to set optional args. <>/<<>> (default:str=None)

        <max_epochs> ... number of epochs to train (default:int = 10)
        <save_epochs> ... save model during these epochs (default:list = [0,1,10,15,....])
        <optim> ... a valid torch optimizer (default = torch.optim.SGD)
        <loss> .... a valid torch loss (default = torch.nn.CrossEntropyLoss)
        <<>> ... arguments for selected torch optimizer, either in <config> or as kwargs
        <<>>  ... arguments for selected torch loss, either in <config> or as kwargs
        **kwargs ... to set optional args. <>/<<>>
    """
    # Configuration
    conf = opts.TrainTorch(config,**kwargs)
    loss_id = getattr(torch.nn,conf.loss)
    conf_loss = opts.PyClass(loss_id, config,**kwargs)
    loss = loss_id(**conf_loss.as_kwargs())
    optim_id = getattr(torch.optim,conf.optim)
    conf_optim = opts.PyClass(optim_id, config,**kwargs)
    optim = optim_id(mdl.parameters(),**conf_optim.as_kwargs())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mdl_final = ''
    loss_final = float("inf")
    log = list()

    # Initial model/info save
    mdl = mdl.to(device)
    fid = path.join(exp_dir,'0.pth')
    torch.save({'epoch':0, 'state_dict':mdl.state_dict()}, fid)
    io.write_scp(path.join(exp_dir,'device'),{'device':device})
    log.append(f'Training {mdl._get_name()}, initial model saved into {fid}.')
    log.append(f'Using {device} to train the network.')

    # Training loop
    for epoch in range(1,conf.max_epochs+1):
        mdl.train()
        loss_dev, loss_train = 0.0, 0.0
        for batch in trainloader:
            sample = batch['sample'].to(device)
            target = batch['target'].long().to(device)

            optim.zero_grad()
            L = loss(mdl(sample),target.reshape(-1))
            L.backward()
            optim.step()
            loss_train += L.item()

        # Dev loss
        mdl.eval()
        for batch in devloader:
            sample = batch['sample'].to(device)
            target = batch['target'].long().to(device)
            L = loss(mdl(sample),target.reshape(-1))
            loss_dev += L.item()

        # Save training checkpoint
        if epoch in conf.save_epochs:
            fid = path.join(exp_dir,str(epoch)+'.pth')
            torch.save({'epoch':epoch,'state_dict':mdl.state_dict()},fid)
            if loss_dev < loss_final:
                loss_final = loss_dev
                mdl_final = fid
            log.append(f'epoch: {epoch}, L_train={round(loss_train,3)}, L_dev={round(loss_dev,3)}')
    os.symlink(mdl_final, f'{exp_dir}/final.pth')

    # Write info into a log
    log.append(f'Training done, final model saved in {exp_dir}/final.pth.')
    with open(path.join(exp_dir,'train_log.txt'), 'w') as log_fid:
        for line in log:
            print(line, file=log_fid)
