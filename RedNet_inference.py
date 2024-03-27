import argparse  # 명령줄 인자를 파싱하기 위한 모듈
import torch  # PyTorch 라이브러리
import imageio  # 이미지 읽기/쓰기를 위한 라이브러리
import skimage.transform  # 이미지 변형을 위한 라이브러리
import torchvision  # PyTorch의 비전 관련 라이브러리

import torch.optim  # 최적화 알고리즘 모듈
import RedNet_model  # RedNet 모델 정의가 포함된 스크립트
from utils import utils  # 유틸리티 함수 모음
from utils.utils import load_ckpt  # 체크포인트 로드 함수

# 명령줄 인자 설정
parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')
parser.add_argument('-r', '--rgb', default=None, metavar='DIR',
                    help='path to image')  # RGB 이미지 경로
parser.add_argument('-d', '--depth', default=None, metavar='DIR',
                    help='path to depth')  # 깊이 이미지 경로
parser.add_argument('-o', '--output', default=None, metavar='DIR',
                    help='path to output')  # 출력 이미지 경로
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')  # CUDA 사용 여부
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  # 모델 체크포인트 경로

args = parser.parse_args()  # 인자 파싱
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")  # 디바이스 설정

image_w = 640  # 이미지 너비
image_h = 480  # 이미지 높이

# 추론 과정
def inference():
    model = RedNet_model.RedNet(pretrained=False)  # 모델 인스턴스 생성
    load_ckpt(model, None, args.last_ckpt, device)  # 모델 체크포인트 로드
    model.eval()  # 평가 모드 설정
    model.to(device)  # 모델을 디바이스로 이동

    image = imageio.imread(args.rgb)  # RGB 이미지 로드
    depth = imageio.imread(args.depth)  # 깊이 이미지 로드

    # 이미지 전처리
    image = skimage.transform.resize(image, (image_h, image_w), order=1, mode='reflect', preserve_range=True)
    depth = skimage.transform.resize(depth, (image_h, image_w), order=0, mode='reflect', preserve_range=True)

    image = image / 255
    image = torch.from_numpy(image).float()
    depth = torch.from_numpy(depth).float()
    image = image.permute(2, 0, 1)
    depth.unsqueeze_(0)

    # 정규화
    image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    depth = torchvision.transforms.Normalize(mean=[19050], std=[9650])(depth)

    image = image.to(device).unsqueeze_(0)
    depth = depth.to(device).unsqueeze_(0)

    pred = model(image, depth)  # 모델 추론

    output = utils.color_label(torch.max(pred, 1)[1] + 1)[0]  # 결과 라벨링

    imageio.imsave(args.output, output.cpu().numpy().transpose((1, 2, 0)))  # 결과 이미지 저장

if __name__ == '__main__':
    inference()  # 추론 실행
