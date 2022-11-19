import os
import logging
import argparse
import datetime
import torch
import os, sys
import pprint

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import Yolo_dataset
from models import Yolov4
from cfg import Cfg

from easydict import EasyDict as edict


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/logs/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nc', '--num_classes', metavar='NC', type=int, nargs='?', default=2,
                        help='Number of Classes', dest='num_classes')
    parser.add_argument('-ch', '--checkpoint_path', metavar='S', type=str, nargs='?', default=None,
                        help='Path for the pretrained weights', dest='checkpoint_path')
    parser.add_argument('-c', '--classses', dest='classes', type=str, default=None,
                        help='Class names')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-l', '--running_locally', dest='running_locally', action='store_true',
                        help='Running on local host')
    parser.set_defaults(running_locally=False)
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)


def test(model, config):
    test_dataset = Yolo_dataset(config.test_label, config)

    n_test = len(test_dataset)

    test_loader = DataLoader(test_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8,
                            pin_memory=True, drop_last=True)

    writer = SummaryWriter(
        log_dir=config.TRAIN_TENSORBOARD_DIR,
        filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
        comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, batch in enumerate(test_loader):
        images = batch[0]
        bboxes = batch[1]

        images = images.to(device=device, dtype=torch.float32)
        bboxes = bboxes.to(device=device)

        bboxes_pred = model(images)

        print(bboxes_pred)


if __name__ == "__main__":
    logging = init_logger(log_dir='logs')
    cfg = get_args(**Cfg)
    print('-' * 99)
    print('Config: ')
    pprint.pprint(cfg)
    print('-' * 99)

    pred_dir = '/content/drive/MyDrive/pytorch-YOLOv4/predictions/'
    if not os.path.exists(os.path.join(pred_dir, 'test')) and not cfg.running_locally:
        os.makedirs(os.path.join(pred_dir, 'test'))

    weight_file = cfg.checkpoint_path

    model = Yolov4(n_classes=cfg.num_classes)
    if not cfg.running_locally:
        model.cuda()
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')

    pretrained_dict = torch.load(weight_file, map_location=map_location)
    model.load_state_dict(pretrained_dict)

    try:
        test(model=model, config=cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)