# 이 코드는 RedNet 모델을 학습시키기 위해 구성되어 있으며, 다음과 같은 주요 기능들을 포함합니다:

# - **데이터 준비**: `DataLoader`와 전처리 트랜스포메이션을 사용하여 학습 데이터를 준비합니다.
# - **모델 설정**: RedNet 모델을 초기화하고, 필요에 따라 DataParallel을 사용하여 멀티 GPU 학습을 설정합니다.
# - **손실 함수와 옵티마이저 정의**: 가중치가 적용된 크로스 엔트로피 손실 함수와 SGD 옵티마이저를 사용합니다.
# - **학습 과정**: 에포크 단위로 모델을 학습시키며, 각 배치에서 손실을 계산하고 역전파를 통해 모델을 업데이트합니다.
# - **진행 상황 로깅과 시각화**: 학습 과정에서 주기적으로 로그를 출력하고, 텐서보드를 통해 학습 과정을 시각화합니다.
# - **체크포인트 저장 및 로딩**: 학습 과정 중에 모델의 상태를 주기적으로 저장하고, 필요시 마지막 체크포인트에서 학습을 재개할 수 있습니다.


# 필수 라이브러리와 모듈을 임포트
import argparse
import os
import time
import torch

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn

from tensorboardX import SummaryWriter

import RedNet_model
import RedNet_data
from utils import utils
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
from torch.optim.lr_scheduler import LambdaLR

# 명령줄 인자를 정의합니다. 학습을 위한 여러 설정을 사용자가 지정할 수 있습니다.
parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to SUNRGB-D')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')

args = parser.parse_args()

# CUDA를 사용할 수 있는 경우 GPU를 사용하도록 설정합니다.
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"
def train():
    # 데이터 전처리와 로딩을 위한 설정을 합니다.
    train_data = RedNet_data.SUNRGBD(transform=transforms.Compose([RedNet_data.scaleNorm(),
                                                                   RedNet_data.RandomScale((1.0, 1.4)),
                                                                   RedNet_data.RandomHSV((0.9, 1.1),
                                                                                         (0.9, 1.1),
                                                                                         (25, 25)),
                                                                   RedNet_data.RandomCrop(image_h, image_w),
                                                                   RedNet_data.RandomFlip(),
                                                                   RedNet_data.ToTensor(),
                                                                   RedNet_data.Normalize()]),
                                     phase_train=True,
                                     data_dir=args.data_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)

    num_train = len(train_data)

    # 모델을 초기화합니다. 필요에 따라 사전 학습된 모델을 로드하거나 새 모델을 사용합니다.
    if args.last_ckpt:
        model = RedNet_model.RedNet(pretrained=False)
    else:
        model = RedNet_moel.RedNet(pretrained=True)
    # 사용 가능한 GPU가 여러 개인 경우, DataParallel을 사용해 모델을 병렬로 학습합니다.
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    # 가중치가 적용된 크로스 엔트로피 손실 함수를 설정합니다.
    CEL_weighted = utils.CrossEntropyLoss2d()

    # 모델과 손실 함수를 지정한 디바이스로 이동합니다.
    model.train()
    model.to(device)
    CEL_weighted.to(device)

    # 옵티마이저를 설정합니다. 여기서는 SGD를 사용합니다.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    # 전역 스텝 변수를 초기화합니다.
    global_step = 0

    # 최신 체크포인트가 있을 경우, 해당 체크포인트에서 학습을 재개합니다.
    if args.last_ckpt:
        global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)

    # 학습률 스케쥴러를 설정합니다. 지정된 에포크마다 학습률이 감소합니다.
    lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    # 텐서보드를 위한 요약(summary) 작성자를 설정합니다.
    writer = SummaryWriter(args.summary_dir)

    # 학습 루프를 시작합니다.
    for epoch in range(int(args.start_epoch), args.epochs):

        scheduler.step(epoch)
        local_count = 0
        last_count = 0
        end_time = time.time()

        # 지정된 주기마다 모델의 체크포인트를 저장합니다.
        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                      local_count, num_train)

        # 배치 데이터를 순회하며 학습을 진행합니다.
        for batch_idx, sample in enumerate(train_loader):
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
            optimizer.zero_grad()
            pred_scales = model(image, depth, args.checkpoint)
            loss = CEL_weighted(pred_scales, target_scales)
            loss.backward()
            optimizer.step()
            local_count += image.data.shape[0]
            global_step += 1

            # 지정된 주기마다 로그를 출력하고, 텐서보드에 정보를 기록합니다.
            if global_step % args.print_freq == 0 or global_step == 1:

                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_log(global_step, epoch, local_count, count_inter,
                          num_train, loss, time_inter)
                end_time = time.time()

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins='doane')
                grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('image', grid_image, global_step)
                grid_image = make_grid(depth[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('depth', grid_image, global_step)
                grid_image = make_grid(utils.color_label(torch.max(pred_scales[0][:3], 1)[1] + 1), 3, normalize=False,
                                       range=(0, 255))
                writer.add_image('Predicted label', grid_image, global_step)
                grid_image = make_grid(utils.color_label(target_scales[0][:3]), 3, normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)
                writer.add_scalar('CrossEntropyLoss', loss.data, global_step=global_step)
                writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=global_step)
                last_count = local_count
                
    # 학습 완료 후, 최종 체크포인트를 저장합니다.
    save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs,
              0, num_train)

    print("Training completed ")

if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    train()
