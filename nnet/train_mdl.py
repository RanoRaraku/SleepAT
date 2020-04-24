"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path
import torch
from torch import optim, cuda, nn
import sleepat
from sleepat import io, opts


def train_mdl(mdl, trainloader, devloader, exp_dir:str, config:str = None,
    **kwargs):
    """
    Train a mdl using the trainset. Currently just a proxy so I
    can delete torch imports from steps. Warning, CE loss needs .long()
    but this is likely not case for other loss functions.
    Arguments:
        mdl ... torch model to train
        trainloader ... torch DataLoader class with train data
        devloader ... torch DataLoader class with development data
        exp_dir ... output directory of training
        <max_epochs> ... number of epochs to train (not used)
        <save_epochs> ... save model during these epochs (not used)
        <<CrossEntropyLoss>> ... see torch.nn.CrossEntropyLoss for args.
        <<SGD>>  ... see torch.optim.SGD for args.
        config ... configuration file to specify all above
        **kwargs ... kwargs for specify all above
    ToDo:
        make SGD and CEL an option        
    """
    # Configuration
    max_epochs = 100
    save_epochs = [1,5,10,15,20,30,40,50,60,70,80,90,100]
    conf_CEL = opts.PyClassOpts(nn.CrossEntropyLoss,config,**kwargs)
    conf_SGD = opts.PyClassOpts(optim.SGD,config,**kwargs)
    device = 'cuda' if cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss(**conf_CEL.as_kwargs())
    optimizer = optim.SGD(mdl.parameters(),**conf_SGD.as_kwargs())
    log = path.join(exp_dir,'log')
    mdl_final = ''
    loss_final = float("inf")

    # Initial model save
    mdl = mdl.to(device)
    fid = path.join(exp_dir,'0.pth')
    torch.save({'epoch': 0, 'train loss': '-',
        'model_state_dict': mdl.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, fid)
    with open(log, 'w') as log_fid:
        print(f'Training DNN, initial model saved into {fid}.', file=log_fid)
    with open(log, 'a') as log_fid:
        print(f'Using {device} to train the network.', file=log_fid)
    io.write_scp(path.join(exp_dir,'device'),{'device':device})

    # Training
    for epoch in range(1,max_epochs+1):
        mdl.train()
        loss_dev, loss_train = 0.0, 0.0
        for batch in trainloader:
            sample = batch['sample'].to(device)
            target = batch['target'].long().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = criterion(mdl(sample),target)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        # Development Set Evaluation
        mdl.eval()
        for batch in devloader:
            sample = batch['sample'].to(device)
            target = batch['target'].long().to(device)

            # forward
            loss = criterion(mdl(sample),target)
            loss_dev += loss.item()

        # Save training checkpoint and write to log
        if epoch in save_epochs:
            fid = path.join(exp_dir,str(epoch)+'.pth')
            torch.save({'epoch': epoch, 'train loss': loss,
                'model_state_dict': mdl.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, fid)
            with open(log, 'a') as log_fid:
                print(f'Model saved into {fid}', file=log_fid)
            with open(log, 'a') as log_fid:
                print(f'Epoch = {epoch}, loss_train = {loss_train}, loss_dev = {loss_dev}.', file=log_fid)
            if loss_dev < loss_final:
                loss_final = loss_dev
                mdl_final = fid 

    os.symlink(mdl_final, f'{exp_dir}/final.pth')
    print(f'Finished training model in {exp_dir}. Final model saved in {exp_dir}/final.pth.')
