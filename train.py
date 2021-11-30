import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datasets
import models
import utils
from numpy import linalg as LA
from scipy.stats import mode
from torch.cuda.amp import autocast as autocast, GradScaler

import einops

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models,datasets
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

import pickle
import torch.utils.data as data

class MiniImageNet(Dataset):
    def __init__(self, setname, args):
        IMAGE_PATH = os.path.join(args.data_dir, 'miniimagenet/images/')
        SPLIT_PATH = os.path.join(args.data_dir, 'miniimagenet/split/')
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1
        self.wnids = []
        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)
        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.num_class = len(set(label))
        if setname == 'val' or setname == 'test':
            image_size = args.image_size
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif setname == 'train':
            image_size = args.image_size
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

## resnet12
def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)

def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)

def norm_layer(planes):
    return nn.BatchNorm2d(planes)

class Block(nn.Module):
    def __init__(self, inplanes, planes, downsample):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.maxpool(out)
        return out

class ResNet12(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.inplanes = 3
        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        self.out_dim = channels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x

def resnet12():
    return ResNet12([64, 128, 256, 512])

##一些常用函数，如分割support和query或者准备label
def split_shot_query(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query

def split_shot_query2(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous()#.view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query

def make_nk_label(n, k, ep_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label

def prepare_label(n_way,n_shot,n_query,ep_per_batch=1):
    # prepare one-hot label
    label = torch.arange(n_way, dtype=torch.int16).repeat(n_query).repeat(ep_per_batch)
    label_aux = torch.arange(n_way, dtype=torch.int8).repeat(n_shot + n_query).repeat(ep_per_batch)
    
    label = label.type(torch.LongTensor)
    label_aux = label_aux.type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()
        label_aux = label_aux.cuda()
    return label, label_aux

##任务采样器
class CategoriesSampler():
    ## 11111 2222 33333 44444 55555 
    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls,
                                           replace=False)
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_per,
                                         replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch) # bs * n_cls * n_per
            yield batch.view(-1)


##所提出的模型
class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dense_1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dense_2(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, token_dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.mlp_token = MlpBlock(token_dim, token_mlp_dim)
        self.mlp_channel = MlpBlock(hidden_dim, channel_mlp_dim)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        y = self.layer_norm_1(x)
        y = y.permute(0,2,1)
        y = self.mlp_token(y)
        y = y.permute(0,2,1)
        x = x + y
        y = self.layer_norm_2(x)
        return x + self.mlp_channel(y)

class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.linear(x)

class Mlp_res12_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet12()
        self.MixerBlock1 = MixerBlock(64,42*42,64,64)
        self.MixerBlock2 = MixerBlock(128,21*21,128,128)
        self.MixerBlock3 = MixerBlock(256,10*10,256,256)
        self.MixerBlock4 = MixerBlock(512,25,512,512)
        self.classifier = LinearClassifier(512,64)

    def forward(self,x):
        x = self.encoder.layer1(x)#torch.Size([32, 64, 42, 42])
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        x = self.MixerBlock1(x)
        x = einops.rearrange(x, 'n (h w) c ->n c h w',h = 42)

        x = self.encoder.layer2(x)#torch.Size([32, 128, 21, 21])
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        x = self.MixerBlock2(x)
        x = einops.rearrange(x, 'n (h w) c ->n c h w',h = 21)

        x = self.encoder.layer3(x)#torch.Size([32, 256, 10, 10])
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        x = self.MixerBlock3(x)
        x = einops.rearrange(x, 'n (h w) c ->n c h w',h = 10)

        x = self.encoder.layer4(x)#torch.Size([32, 512, 5, 5])
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        x = self.MixerBlock4(x)
        x = einops.rearrange(x, 'n (h w) c ->n c h w',h = 5)

        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        #x = self.classifier(x)
        return x

#metabaseline ： https://arxiv.org/pdf/2003.04390.pdf

class MetaBaseline(nn.Module):
    def __init__(self, encoder, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = encoder  ### 进来的resnet12需要去掉全连接层和gap使其输出为32,512,5,5
        self.method = method
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp
        self.laten_dim = 512

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]
        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0)) ##torch.Size([320, 512])
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits


def main():
    svname = args.name
    if svname is None:
        svname = 'meta_Mlp_Res12_{}_{}_{}_{}-{}shot'.format(args.image_size,args.lr,args.ep_per_batch,
                args.train_dataset, args.n_shot)

    save_path = os.path.join('./save', svname)
    test_time = 0  # 这个用于保存不同批次的测试文件，简称文件名
    path = save_path+f"{test_time}"
    while os.path.exists(path):
        test_time +=1
        path = save_path+f"{test_time}"
    save_path = path
    print(save_path)
    print(args.ep_per_batch)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    utils.log('n_way:{}\n n_shot:{}\n task_batch:{}\n train_batches: {}\n test_batches:{}\n '
            'lr:{} \n  weight_decay:{}\n image_size:{} testbatches{}\n'.format(args.n_way,args.n_shot,args.ep_per_batch,
            args.train_batches,10000,args.lr,args.weight_decay,args.image_size,args.test_batches))
    n_way, n_shot = args.n_way, args.n_shot
    n_query = args.n_query
    n_train_way = args.n_way
    n_train_shot = args.n_shot
    ep_per_batch = args.ep_per_batch
    train_dataset = MiniImageNet('train',args) 
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),64))
    if args.visualize_datasets:
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    train_sampler = CategoriesSampler(
            train_dataset.label, args.train_batches,
            n_train_way,#*args.ep_per_batch,
            n_train_shot + n_query,args.ep_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True) ##训练的训练集的采样
    tval_dataset = MiniImageNet('test', args)
    utils.log('tval dataset: {} (x{}), {}'.format(
            tval_dataset[0][0].shape, len(tval_dataset),20))
    if args.visualize_datasets:
        utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
    tval_sampler = CategoriesSampler(
            tval_dataset.label, args.train_batches,
            n_way, n_shot + n_query,args.ep_per_batch)
    tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                num_workers=args.num_workers, pin_memory=True) ## 训练过程中测试集的采样
    test_sampler = CategoriesSampler(
            tval_dataset.label, args.test_batches,
            n_way, n_shot + n_query,args.ep_per_batch)
    test_loader = DataLoader(tval_dataset, batch_sampler=test_sampler,
                                num_workers=args.num_workers, pin_memory=True)  ## 最后一万个任务的测试集采样
    val_dataset  = MiniImageNet('val',args)
    utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),16))
    if args.visualize_datasets:
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
            val_dataset.label,args.train_batches,
            n_way, n_shot + n_query,args.ep_per_batch)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)  ## 训练过程中验证集的采样


    #### Model and optimizer ####
    ### 设置模型和参数 ########
    encoder = Mlp_res12_classifier().cuda() ## encoder可以为resnet12，预训练参数也要变为resnet12的
    #加载预训练参数
    encoder.load_state_dict(torch.load('/home/zxz/code/meta-baseline/pre_Mlp_Res12/max_va_mlp_res12_classifier_512_size_84_75.7221.pth')) ##改成你的地址
    # encoder = resnet12().cuda()
    # encoder.load_state_dict(torch.load('/home/zxz/code/meta-baseline/resnet_few_shot/max_va_resnet12_512_76.0417.pth'))

    model = MetaBaseline(encoder).cuda()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    optimizer  = torch.optim.SGD(model.parameters(),lr = args.lr,momentum=0.9,weight_decay = args.weight_decay) #优化器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 5, gamma=0.1, last_epoch=-1) #学习率变化
    ########
    max_epoch = args.max_epoch
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}
        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        np.random.seed(epoch)
        train_gen = tqdm(train_loader)
        k = args.n_way * args.n_shot
        for i,(data,l) in enumerate(train_gen, 1):
            x_shot, x_query = split_shot_query(data.cuda(), n_train_way, n_train_shot, n_query,
                            ep_per_batch=ep_per_batch)
            label = make_nk_label(n_train_way, n_query,ep_per_batch=ep_per_batch).cuda()
            with autocast():
                logits = model(x_shot, x_query).view(-1, n_train_way)
                loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc2(logits, label)*100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            aves['tl'].add(loss.item())
            aves['ta'].add(acc)
            train_gen.set_description('训练阶段:epo {} 平均loss1={:.4f}  平均acc={:.4f}'.format(epoch, aves['tl'].item(),aves['ta'].item()))
            logits = None; loss = None 
        aveacc = aves['ta'].item()
        # eval
        model.eval()
        for name, loader, name_l, name_a in [
                ('tval', tval_loader, 'tvl', 'tva'),
                ('val', val_loader, 'vl', 'va')]:
            np.random.seed(0)
            loader_gen = tqdm(loader)
            for i, (data,l) in enumerate(loader_gen, 1):
                x_shot, x_query = split_shot_query(
                    data.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=ep_per_batch)  
                label = make_nk_label(n_train_way, n_query,ep_per_batch=args.ep_per_batch).cuda()
                with torch.no_grad():
                    logits = model(x_shot, x_query).view(-1, n_way)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc2(logits, label)*100          
                aves[name_l].add(loss.item())
                aves[name_a].add(acc)
                loader_gen.set_description('{}阶段: epo {} 平均loss1={:.4f}  平均acc={:.4f}'.format(name, epoch, aves[name_l].item(),aves[name_a].item()))
        # post
        lr_scheduler.step()
        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])
        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                'val {:.4f}|{:.4f}, {} {}/{} \n'.format(
                epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
                aves['vl'], aves['va'], t_epoch, t_used, t_estimate))
        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)    
        training = {
            'epoch': epoch,
            # 'optimizer': args.optimizer,
            # 'optimizer_args': args.optimizer_args,
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            #'config': config,
            'model': args.model,
            'model_args': args.encoder,
            'model_sd': model.state_dict(),
            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        '''
        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
        '''
        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max_mlp_res12.pth'))
            torch.save(model.encoder.state_dict,os.path.join(save_path, 'max_encoder_mlp_res12.pth'))
        writer.flush()
    #######    训练结束，开始最后测试  ###########
    model.load_state_dict(torch.load(save_path+'/max_mlp_res12.pth')['model_sd'])
    model.eval()
    tl = utils.Averager()
    ta = utils.Averager()
    k = args.n_way * args.n_shot
    np.random.seed(0)
    test_gen = tqdm(test_loader)
    for i, (data,l) in enumerate(test_gen, 1):
        data = data.cuda()
        x_shot, x_query = split_shot_query(
                    data.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=ep_per_batch)  
        label = make_nk_label(n_train_way, n_query,ep_per_batch=args.ep_per_batch).cuda()       
        with torch.no_grad():
            logits = model(x_shot, x_query).view(-1, n_way)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc2(logits, label)*100     
        tl.add(loss.item())
        ta.add(acc)
        test_gen.set_description('测试阶段（10000）:  平均loss1={:.4f}  平均acc={:.4f}'.format(tl.item(),ta.item()))
    utils.log('最后{}个小样本任务的测试结果： loss={:.4f}   acc={:.4f} '.format(10000,tl.item(),ta.item()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('-image_size', type=int, default=84)
    parser.add_argument('-n_way', type=int, default=5)
    parser.add_argument('-n_shot', type=int, default=1)
    parser.add_argument('-n_query', type=int, default=15,help='number of query image per class')
    parser.add_argument('-train_batches', type=int, default=200)
    parser.add_argument('-ep_per_batch', type=int, default=2)
    parser.add_argument('-num_workers', type=int, default=24)
    parser.add_argument('-max_epoch', type=int, default=20)
    parser.add_argument('-visualize_datasets', type=bool, default=False)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=7e-4)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-test_batches', type=int, default=5000)
    parser.add_argument('-train_dataset', type=str, default='mini-imagenet')
    parser.add_argument('-train_dataset_args', default={'split':'train'})
    parser.add_argument('-tval_dataset_TF', type=bool, default=True)
    parser.add_argument('-tval_dataset', type=str, default='mini-imagenet')
    parser.add_argument('-tval_dataset_args', default={'split':'test'})
    parser.add_argument('-val_dataset', type=str, default='mini-imagenet')
    parser.add_argument('-val_dataset_args', default={'split':'val'})
    parser.add_argument('-model_args', default={'encoder': 'resnet12','encoder_args': {}})
    parser.add_argument('-data_dir', type=str, default='/home/zxz/code/meta-baseline/materials')
    parser.add_argument('-model', type=str, default='res12_vit')
    parser.add_argument('-encoder', type=str, default='resnet12')
    args = parser.parse_args()
    utils.set_gpu(args.gpu)
    
    args.n_shot = 1
    main()
    args.n_shot = 5
    main()
