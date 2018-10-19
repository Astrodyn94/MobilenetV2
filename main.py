from dataloader import *
from option.default_option import TrainOptions
from model import * 
import torch
import torch.nn as nn 
from torch.optim import RMSprop,Adam,SGD
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import warnings
import tqdm
import scipy.misc
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")

opt = TrainOptions()
writer = SummaryWriter('runs/' + opt.name )

if opt.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    device = torch.device('cuda')

trainSet = caltech_dataloader(opt , phase = 'train')
train_loader = DataLoader(trainSet , batch_size= opt.batch_size , num_workers = 8,shuffle = True , pin_memory = False)

testSet = caltech_dataloader(opt , phase = 'val')
test_loader = DataLoader(testSet , batch_size= opt.batch_size , num_workers = 8,shuffle = True , pin_memory = False)

model = MobileNetV2(opt)
if opt.multi_gpu:
    model = torch.nn.DataParallel(model)
    model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
optim= SGD(model.parameters(), lr= opt.lr)
lr_scheduler = StepLR(optim , step_size = 1 , gamma = opt.gamma)
f = open(opt.name + '.txt' , "w")
f.write("i  train_loss  train_acc   test_loss   test_acc\n")

best_acc = 0
for i in range(opt.niter):
    f.write("%d "%i)
    train_losses = [];test_losses=[]
    corrects = 0
    for k,item in tqdm.tqdm(enumerate(train_loader,1)):
        label,name,inputs = item

        inputs = Variable(inputs).cuda().float(); target = Variable(label , requires_grad = False).cuda().long()
        output = model(inputs)
        _ , preds = torch.max(output.data , 1)
        corrects +=torch.sum(preds == target)
        
        ##### train model #####
        optim.zero_grad()
        loss = criterion(output , target) 
        loss.backward()
        optim.step()
        lr_scheduler.step()
        train_losses.append(loss.data)
    
    writer.add_scalar('data/train_loss' , np.sum(train_losses) / len(train_losses),i)
    writer.add_scalar('data/train_acc' , float(corrects) / len(trainSet) ,i)
    f.write("%5.5f  "% (np.sum(train_losses) / len(train_losses)))
    f.write("%5.5f  "% (corrects / len(trainSet)))
    print('i : ',i)
    print('train loss' ,  np.sum(train_losses) / len(train_losses))
    print('train acc' ,  float(corrects) / len(trainSet))

    corrects = 0
    for k,item in tqdm.tqdm(enumerate(test_loader,1)):
        label,name,inputs = item

        inputs = Variable(inputs).cuda().float(); target = Variable(label , requires_grad = False).cuda().long()
        output = model(inputs)
        _ , preds = torch.max(output.data , 1)
        corrects +=torch.sum(preds == target)

        loss = criterion(output , target) 
        test_losses.append(loss.data)

    test_acc = float(corrects) / len(testSet)
    writer.add_scalar('data/test_loss' , np.sum(test_losses) / len(test_losses),i)
    writer.add_scalar('data/test_acc' , test_acc ,i)
    print('test loss' ,  np.sum(test_losses) / len(test_losses))
    print('test acc' ,  test_acc)
    
    if test_acc > best_acc:
        save_model(model,i,opt)
        best_acc = test_acc

    f.write("%5.5f  "% (np.sum(test_losses) / len(test_losses)))
    f.write("%5.5f  "% (float(corrects) / len(testSet)) ) 
    f.write("\n")
f.close()