import time
import torch
import tqdm

from utils.eval_utils import accuracy
import numpy as np 

__all__ = ["train", "validate", "modifier"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    losses = []
    top1 = []
    top5 = []
    n_data_points = []
   

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    new_lam = 0.5*(1+np.cos(np.pi*(epoch)/(args.epochs)))*args.lamb
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
       

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target,  new_lam, epoch, True)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.append(loss.item())
        top1.append(acc1.item())
        top5.append(acc5.item())
        n_data_points.append(images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            
    top1 = np.array(top1)
    top5 = np.array(top5)
    n_data_points = np.array(n_data_points)
    top1 = sum(top1*n_data_points)/sum(n_data_points)
    top5 = sum(top5*n_data_points)/sum(n_data_points)
    losses = sum(losses*n_data_points)/sum(n_data_points)
    return top1, top5, losses



def validate(val_loader, model, criterion, args, writer, epoch):
    losses = []
    top1 = []
    top5 = []
    n_data_points = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target, None, epoch, False)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.append(loss.item())
            top1.append(acc1.item())
            top5.append(acc5.item())
            n_data_points.append(images.size(0))

            # measure elapsed time
           
            end = time.time()

           

    top1 = np.array(top1)
    top5 = np.array(top5)
    losses = np.array(losses)
    n_data_points = np.array(n_data_points)
    top1 = sum(top1*n_data_points)/sum(n_data_points)
    top5 = sum(top5*n_data_points)/sum(n_data_points)
    losses = sum(losses*n_data_points)/sum(n_data_points)
    return top1, top5, losses

def modifier(args, epoch, model):
    return


def get_output(val_loader, model, criterion, args, writer, epoch):

    # switch to evaluate mode
    model.eval()
    all_outputs = []
    all_gts = []
    all_losses = []
    losses = []
    top1 = []
    top5 = []
    n_data_points = []
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
           
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            
            all_gts.extend(target.data.cpu().numpy().flatten())
            all_outputs.extend(output.data.cpu().numpy())

            # measure accuracy and record loss
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.append(acc1.item())
            top5.append(acc5.item())
            n_data_points.append(images.size(0))

            # measure elapsed time

            end = time.time()
            

    all_gts = np.array(all_gts)
    all_outputs = np.array(all_outputs)    
    top1 = np.array(top1)
    top5 = np.array(top5)
    losses = np.array(losses)
    n_data_points = np.array(n_data_points)
    top1 = sum(top1*n_data_points)/sum(n_data_points)
    print("Top 1 Accuracy is", top1)
    top5 = sum(top5*n_data_points)/sum(n_data_points)
    


    return top1, top5, all_gts, all_losses, all_outputs 