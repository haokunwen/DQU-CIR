import os 
import argparse
import logging
import warnings 
import random

import numpy as np 
import torch 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import dataloader
from tqdm import tqdm

import open_clip 
import utils
import datasets
import model
import test

from torch.cuda.amp import autocast as autocast, GradScaler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--dataset', default = 'dress', help = "data set type")
parser.add_argument('--fashioniq_split', default = 'val-split')
parser.add_argument('--fashioniq_path', default = '../data/FashionIQ')
parser.add_argument('--shoes_path', default = '../data/Shoes')
parser.add_argument('--fashion200k_path', default = '../data/Fashion200K')
parser.add_argument('--cirr_path', default = '../data/CIRR')

parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--hidden_dim', type=int, default=1024)

parser.add_argument('--seed', type=int, default=42)   
parser.add_argument('--lr', type=float, default=1e-4) 
parser.add_argument('--clip_lr', type=float, default=1e-6) 

parser.add_argument('--backbone', type=str, default='ViT-H-14')

parser.add_argument('--lr_decay', type=int, default=8)
parser.add_argument('--lr_div', type=float, default=0.1)  
parser.add_argument('--max_decay_epoch', type=int, default=10)  
parser.add_argument('--tolerance_epoch', type=int, default=5)

 
parser.add_argument('--model_dir', default='./checkpoints',
                    help="Directory containing params.json")

parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--i', type=str, default='0')

args = parser.parse_args()


def load_dataset():
    # _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14', pretrained=os.path.join('../models/laionCLIP-ViT-H-14-laion2B-s32B-b79K', 'open_clip_pytorch_model.bin'))
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')
    if args.dataset in ['dress', 'shirt', 'toptee']:
        print('Loading FashionIQ-{} dataset'.format(args.dataset))
        fashioniq_dataset = datasets.FashionIQ(path = args.fashioniq_path, category = args.dataset, transform = [preprocess_train, preprocess_val], split = args.fashioniq_split)
        return [fashioniq_dataset]
    elif args.dataset == 'shoes':
        print('Reading shoes')
        shoes_dataset = datasets.Shoes(path = args.shoes_path, transform = [preprocess_train, preprocess_val])
        return [shoes_dataset]
    elif args.dataset == 'cirr':
        print('Reading cirr')
        cirr_dataset = datasets.CIRR(path = args.cirr_path, transform = [preprocess_train, preprocess_val])
        return [cirr_dataset]
    elif args.dataset == 'fashion200k':
        print('Reading fashion200k')
        fashion200k_dataset = datasets.Fashion200k(path = args.fashion200k_path, split = 'train', transform = [preprocess_train, preprocess_val])
        fashion200k_testset = datasets.Fashion200k(path = args.fashion200k_path, split = 'test', transform = [preprocess_train, preprocess_val])
        return [fashion200k_dataset, fashion200k_testset]



def set_bn_eval(m): 
    classname = m.__class__.__name__ 
    if classname.find('BatchNorm2d') != -1: 
        m.eval() 

def create_model_and_optimizer():
    DQU_CIR_model = model.DQU_CIR(args.hidden_dim, args.dropout_rate)
    DQU_CIR_model.cuda()

    params = list(DQU_CIR_model.named_parameters())
    param_group = [
        {'params': [p for n, p in params if any(nd in n for nd in ['clip'])], 'lr': args.clip_lr},
        {'params': [p for n, p in params if not any(nd in n for nd in ['clip'])], 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(param_group, lr=args.lr, weight_decay = args.weight_decay)
    return DQU_CIR_model, optimizer


def train(model, optimizer, dataloader, scaler):
    model.train()
    model.apply(set_bn_eval)
    summ = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        for i, data in enumerate(dataloader):

            target_img = data['target_img_data'].cuda()
            textual_query = data['textual_query']
            visual_query = data['visual_query'].cuda()
            optimizer.zero_grad()
            with autocast():
                loss = model.compute_loss(textual_query, visual_query, target_img)
                total_loss = loss['ranking']

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()     

            if i % args.save_summary_steps == 0:
                summary_batch = {}
                summary_batch['total_loss'] = total_loss.item()
                summ.append(summary_batch)
            loss_avg.update(total_loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()


def train_and_evaluate(model, optimizer, dataset_list):
    if args.dataset == 'fashion200k':
        fashion200k_testset = dataset_list.pop(-1)
    trainloader = dataloader.DataLoader(dataset_list[0],
                                batch_size = args.batch_size,
                                shuffle = True,
                                num_workers=args.num_workers)


    best_score = float('-inf')
    tolerance = 0
    scaler = GradScaler()
    epoches = args.num_epochs

    for epoch in range(epoches):

        tolerance = tolerance + 1
        if epoch != 0 and (epoch+1) % args.lr_decay == 0 and epoch < args.max_decay_epoch:
            for g in optimizer.param_groups:
                g['lr'] *= args.lr_div

        logging.info("Epoch {}/{}".format(epoch + 1, epoches))
        train(model, optimizer, trainloader, scaler)

        current_score = 0
        if tolerance < args.tolerance_epoch:
            if args.dataset in ['dress', 'shirt', 'toptee', 'shoes']:
                with torch.no_grad():
                    t = test.test(args, model, dataset_list[0], args.dataset)
                logging.info(t)
                current_score = current_score + t[1][1] + t[2][1]

            
            elif args.dataset in ['fashion200k']:
                t = test.test_fashion200k_dataset(args, model, fashion200k_testset)
                logging.info(t)
                current_score = current_score + t[0][1] + t[1][1] + t[2][1]
            
            elif args.dataset in ['cirr']:
                t = test.test_cirr_valset(args, model, dataset_list[0])
                logging.info(t)
                current_score = current_score + t[0][1] + t[1][1] + t[2][1] + t[3][1] + t[4][1] + t[5][1] + t[6][1] # mean best
            
            if current_score > best_score:
                best_score = current_score
                tolerance = 0
                best_json_path = os.path.join(
                    args.model_dir, "{}_{}_metrics_best.json".format(args.dataset, args.i))
                test_metrics = {}
                for metric_name, metric_value in t:
                    test_metrics[metric_name] = metric_value

                utils.save_dict_to_json(test_metrics, best_json_path)
                # save model
                if args.dataset == 'cirr':
                    torch.save(model, os.path.join(args.model_dir, "{}_{}_best_model.pt".format(args.dataset, args.i)))
        else:
            break


if __name__ == '__main__':

    # Load the parameters from json file

    print('Arguments:')
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))

    # seed = args.seed
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  
    # np.random.seed(seed)  # Numpy module.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    utils.set_logger(os.path.join(args.model_dir, '{}_{}_train.log'.format(args.dataset, args.i)))
    logging.info('Loading the datasets and model...')
    # fetch dataloaders

    dataset_list = load_dataset()
 
    model, optimizer = create_model_and_optimizer()
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, optimizer, dataset_list)
