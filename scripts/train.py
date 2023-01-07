import argparse
from tqdm import tqdm
import os

import torch as th
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable

from torchvision import utils

from dataset import *
from losses import *
from model import *
 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/dataset/dataset', help='Directory of Cityscapes dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--loss_type', type=str, default="dice", help='Choose between dice & ce')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=0.005,
                        help='Learning rate', dest='lr')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=19, help='Number of classes')
    parser.add_argument('--num_filters', type=int, default=64, help='Number of filters')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of channels of input')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--augment', default=False, help='Augment image input')
    parser.add_argument('--resume_step', type=int, default=0, help='Step to resume')
    parser.add_argument('--model', type=str, default="unet", help='Choose unet or ')
    
    return parser.parse_args()

    
if __name__ == '__main__':
    if not os.path.exists('./result/'):
        os.makedirs('./result/')
                    
    if not os.path.exists('./chkpt/'):
        os.makedirs('./chkpt/')

    file_loss = open('./chkpt/loss_batch.txt', 'w')
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    args = get_args()
    if args.model == "unet":
        model = UNet(n_channels=args.num_channels, 
                    n_classes=args.num_classes, 
                    n_filters=args.num_filters, 
                    bilinear=args.bilinear)
    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.resume_step != 0:
        try:
            model.load_state_dict(th.load('./chkpt/model_epoch_{}.pt'.format(args.resume_step)))
            optimizer.load_state_dict(th.load('./chkpt/optim_epoch_{}.pt'.format(args.resume_step)))
        except Exception as err:
            print("Can not find the checkpoint")
            raise

    if args.loss_type == "dice":
        loss_func = DiceLoss(ignore_index=255)
    elif args.loss_type == "ce":
        loss_func = CELoss(ignore_index=255)
    
    # 1. Load data
    img_data = CityscapesDataset(args.data_dir, split='train', mode='fine', augment=args.augment)
    img_batch = DataLoader(img_data, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # 2.Training
    with tqdm(range(args.resume_step,args.resume_step+args.epochs+1)) as pbar:
        for i in pbar:
            losses = []
            for idx_batch, (image_rgb, label_mask, label_rgb) in tqdm(enumerate(img_batch)):

                # zero the grad of the network before feed-forward
                optimizer.zero_grad()
                # send to the GPU and do a forward pass
                x = Variable(image_rgb).to(device)
                y_truth = Variable(label_mask).to(device)
                y = model.forward(x)

                # we "squeeze" the groundtruth if we are using cross-entropy loss
                # this is because it expects to have a [N, W, H] image where the values
                # in the 2D image correspond to the class that that pixel should be 0 < pix[u,v] < classes
                y_truth = th.squeeze(y_truth)

                # finally calculate the loss and back propagate
                loss = loss_func(y, y_truth)
                # file_loss.write(str(loss.item())+"\n")
                losses.append(loss)
                loss.backward()
                optimizer.step()

                # every 400 batches, save the current images
                if idx_batch % 400 == 0:
                    # max over the classes should be the prediction
                    # our prediction is [N, classes, W, H]
                    # so we max over the second dimension and take the max response
                    # if we are doing rgb reconstruction, then just directly save it to file
                    y_threshed = th.zeros((y.size()[0], 3, y.size()[2], y.size()[3]))
                    for idx in range(0, y.size()[0]):
                        max_index = th.argmax(y[idx], dim=0).cpu().int()
                        y_threshed[idx] = img_data.class_to_rgb(max_index)
                    # save the original image, label and prediction batches to file
                    img = []
                    img.append(x.cpu().data)
                    img.append(label_rgb/255)
                    img.append(y_threshed/255)
                    img = th.cat(img, dim=0)
                    img = utils.make_grid(img, nrow = 4)
                    utils.save_image(img, "./result/image_{}_{}.png".format(i, idx_batch))
                
            mean_loss = sum(losses) / len(losses)
            print("Epoch = "+str(i)+" | Loss = "+str(mean_loss))
            file_loss.write(str(mean_loss)+"\n")
            # finally save checkpoint each 5 epochs
            if i % 5 == 0:
                th.save(model.state_dict(), './chkpt/model_epoch_{}.pt'.format(i))
                th.save(optimizer.state_dict(), './chkpt/optim_epoch_{}.pt'.format(i))
