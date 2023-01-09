from tqdm import tqdm
import os

import torch as th
import logging

from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import utils

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
    # parser.add_argument('--augment', default=False, help='Augment image input')
    parser.add_argument('--model', type=str, default="unet", help='Choose unet or ...')
    parser.add_argument('--result_folder', type=str, default='./predict/', help='Save results to folder')
    parser.add_argument('--chkpt_file', type=str, default='./chkpt/model_epoch_50.pt', help='File checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    args = get_args()
    if args.model == "unet":
        model = UNet(n_channels=args.num_channels, 
                    n_classes=args.num_classes, 
                    n_filters=args.num_filters, 
                    bilinear=args.bilinear)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(th.load(args.chkpt_file))
    logger.info("Loaded {}".format(args.chkpt_file))

    logger.info("Loading validation data")
    val_data = CityscapesDataset(args.data_dir, split='val', mode='fine', augment=False)
    val_batch = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    with tqdm(enumerate(val_batch)) as pbar:
        for idx_batch, (image_rgb, _, label_rgb) in pbar:
            # send to the GPU and do a forward pass
            x = Variable(image_rgb).to(device)
            y = model.forward(x)
            
            # max over the classes should be the prediction
            # our prediction is [N, classes, W, H]
            # so we max over the second dimension and take the max response
            pred_class = th.zeros((y.size()[0], 3, y.size()[2], y.size()[3]))
            for idx in range(0, y.size()[0]):
                max_index = th.argmax(y[idx], dim=0).cpu().int()
                pred_class[idx] = val_data.class_to_rgb(max_index)

            img = []
            img.append(x.cpu().data)
            img.append(label_rgb/255)
            img.append(pred_class/255)
            img = th.cat(img, dim=0)
            img = utils.make_grid(img, nrow = 4)
            utils.save_image(img, args.result_folder+"image_{}.png".format(idx_batch))