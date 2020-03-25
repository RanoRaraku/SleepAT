"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
from os import symlink
from os.path import join
from torch import save
from torch.optim import SGD
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from sleepat.base.opts import PyClassOpts


def train_mdl(mdl, trainloader, devloader, exp_dir:str,
    config:str = None, **kwargs):
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
    """
    # Configuration
    max_epochs = 100
    save_epochs = [1,5,10,15,20,30,40,50,60,70,80,90,100]
    conf_CEL = PyClassOpts(CrossEntropyLoss,config,**kwargs)
    conf_SGD = PyClassOpts(SGD,config,**kwargs)
    device = 'cuda' if is_available() else 'cpu'
    mdl = mdl.to(device)
    criterion = CrossEntropyLoss(**conf_CEL.as_kwargs())
    optimizer = SGD(mdl.parameters(),**conf_SGD.as_kwargs())
    dev_samples_num = devloader.dataset.sample_size
    log = join(exp_dir,'log')
    final_mdl = ''
    final_acc = 0.0

    # Initial model save
    fid = join(exp_dir,'0.pth')
    save({'epoch': 0, 'train_loss': '-',
        'model_state_dict': mdl.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, fid)
    with open(join(exp_dir,'device'), 'w') as device_fid:
        print(device, file=device_fid)
    with open(log, 'w') as log_fid:
        print(f'Training DNN, initial model saved into {fid}', file=log_fid)


    # Training
    for epoch in range(1,max_epochs+1):
        mdl.train()
        dev_acc, dev_loss, train_loss = 0.0, 0.0, 0.0
        for batch in trainloader:
            sample = batch['sample'].to(device)
            target = batch['target'].long().to(device)

            # Forward, backward, update
            optimizer.zero_grad()
            loss = criterion(mdl(sample),target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Dev set evaluation
        mdl.eval()
        for batch in devloader:
            sample = batch['sample'].to(device)
            target = batch['target'].long().to(device)

            pred = mdl(sample)
            loss = criterion(pred,target)
            dev_loss += loss.item()
            post = softmax(pred,dim=1)
            (_, arg_max) = post.max(dim=1)
            dev_acc += float((arg_max==target).sum())
        dev_acc = round(dev_acc*100/dev_samples_num,3)

        # Save training checkpoint and write to log
        if epoch in save_epochs:
            fid = join(exp_dir,str(epoch)+'.pth')
            save({'epoch': epoch, 'train_loss': loss,
                'model_state_dict': mdl.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, fid)
            with open(log, 'a') as log_fid:
                print(f'Model saved into {fid}', file=log_fid)
            with open(log, 'a') as log_fid:
                print(f'Epoch = {epoch}, train_loss = {train_loss}, dev_loss = {dev_loss}, dev_acc = {dev_acc}.', file=log_fid)
            if dev_acc > final_acc:
                final_acc = dev_acc
                final_mdl = fid 

    symlink(final_mdl, f'{exp_dir}/final.pth')
    print(f'Finished training model in {exp_dir}. Final model saved in {exp_dir}/final.pth.')
