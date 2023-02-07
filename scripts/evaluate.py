from tqdm import tqdm
import argparse
import logging

import torch as th
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from evaluate_metric.metric import *
from dataset.cityscapes import CityscapesDataset
from model.unet import UNet

# Logging
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)

stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
streamformat = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
stream.setFormatter(streamformat)
logger.addHandler(stream)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/dataset/dataset', help='Directory of Cityscapes dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_classes', type=int, default=19, help='Number of classes')
    parser.add_argument('--num_filters', type=int, default=64, help='Number of filters')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of channels of input')
    parser.add_argument('--bilinear', default=False, help='Use bilinear upsampling')
    parser.add_argument('--resnet', default=False, help='Use resnet')
    parser.add_argument('--attention', default=False, help='Use attention')
    # parser.add_argument('--augment', default=False, help='Augment image input')
    parser.add_argument('--model', type=str, default="unet", help='Choose unet or ...')
    parser.add_argument('--model_path', type=str, default='./chkpt/model_epoch_50.pt', help='Path to model')
    return parser.parse_args()

if __name__ == '__main__':
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    args = get_args()
    if args.model == "unet":
        model = UNet(n_channels=args.num_channels,
                     n_classes=args.num_classes,
                     n_filters=args.num_filters,
                     bilinear=args.bilinear,
                     resnet=args.resnet,
                     attention=args.attention)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(th.load(args.model_path))
    logger.info("Loaded {}".format(args.model_path))

    logger.info("Loading validation data")
    val_data = CityscapesDataset(args.data_dir, split='val', mode='fine', augment=False)
    val_batch = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=2)

    
    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = 0,0,0,0
    with tqdm(enumerate(val_batch)) as pbar:
        for idx_batch, (image_rgb, label_mask, label_rgb) in pbar:
            # send to the GPU and do a forward pass
            x = Variable(image_rgb).to(device)
            y_truth = Variable(label_mask)
            y = model.forward(x)
            pred_class = th.argmax(y, dim=1).cpu().int()

            area_intersect, area_union, area_pred_label, area_label = \
                intersect_and_union(pred_class,y_truth,num_classes=args.num_classes,ignore_index=255)
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label
    dic = total_area_to_metrics(total_area_intersect, total_area_union, \
          total_area_pred_label, total_area_label,metrics = ['mIoU', 'mDice', 'mFscore'])
    for metric, value in dic.items():
        logger.info(metric + ": " + str(value.mean().item()))