# 필수 라이브러리 및 모듈 임포트
import numpy as np
import scipy.io
import imageio
import h5py
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
from RedNet_train import image_h, image_w # RedNet_train 스크립트에서 이미지의 높이와 너비를 임포트

# 데이터 파일 경로 정의
img_dir_train_file = './data/img_dir_train.txt'
depth_dir_train_file = './data/depth_dir_train.txt'
label_dir_train_file = './data/label_train.txt'
img_dir_test_file = './data/img_dir_test.txt'
depth_dir_test_file = './data/depth_dir_test.txt'
label_dir_test_file = './data/label_test.txt'

# SUNRGBD 데이터셋을 위한 PyTorch Dataset 클래스 정의
class SUNRGBD(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None):

        self.phase_train = phase_train # 학습 또는 테스트 단계 여부
        self.transform = transform # 적용할 전처리 트랜스폼

        try:
            # 이미지, 깊이, 레이블 파일 경로를 텍스트 파일에서 로드
            with open(img_dir_train_file, 'r') as f:
                self.img_dir_train = f.read().splitlines()
            with open(depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            with open(img_dir_test_file, 'r') as f:
                self.img_dir_test = f.read().splitlines()
            with open(depth_dir_test_file, 'r') as f:
                self.depth_dir_test = f.read().splitlines()
            with open(label_dir_test_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
        except:
            # 텍스트 파일이 없는 경우 SUNRGB-D 데이터셋에서 직접 로드
            if data_dir is None:
                data_dir = '/path/to/SUNRGB-D'
            # SUNRGBD 메타데이터 및 분할 정보 로드
            SUNRGBDMeta_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
            allsplit_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
            SUNRGBD2Dseg_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []
            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []
            self.SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

            SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                           struct_as_record=False)['SUNRGBDMeta']
            split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
            split_train = split['alltrain']

            seglabel = self.SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

            # 이미지, 깊이, 레이블 경로를 메타데이터에서 추출하여 배열에 저장
            for i, meta in enumerate(SUNRGBDMeta):
                # 경로 처리 및 레이블 파일 생성
                # train/test 분할에 따라 데이터 배열에 추가
                meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
                real_dir = meta_dir.replace('/n/fs/sun3d/data', data_dir)
                depth_bfx_path = os.path.join(real_dir, 'depth_bfx/' + meta.depthname)
                rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

                label_path = os.path.join(real_dir, 'label/label.npy')

                if not os.path.exists(label_path):
                    os.makedirs(os.path.join(real_dir, 'label'), exist_ok=True)
                    label = np.array(self.SUNRGBD2Dseg[seglabel[i][0]][()]).transpose(1, 0)
                    np.save(label_path, label)

                if meta_dir in split_train:
                    self.img_dir_train = np.append(self.img_dir_train, rgb_path)
                    self.depth_dir_train = np.append(self.depth_dir_train, depth_bfx_path)
                    self.label_dir_train = np.append(self.label_dir_train, label_path)
                else:
                    self.img_dir_test = np.append(self.img_dir_test, rgb_path)
                    self.depth_dir_test = np.append(self.depth_dir_test, depth_bfx_path)
                    self.label_dir_test = np.append(self.label_dir_test, label_path)
            
            # 분할 정보를 텍스트 파일에 저장
            local_file_dir = '/'.join(img_dir_train_file.split('/')[:-1])
            if not os.path.exists(local_file_dir):
                os.mkdir(local_file_dir)
            with open(img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(label_dir_test_file, 'w') as f:
                f.write('\n'.join(self.label_dir_test))

    def __len__(self):
        # 데이터셋의 길이를 반환합니다.
        if self.phase_train:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        # 데이터셋의 특정 인덱스에 해당하는 샘플을 로드하고 반환합니다.
        if self.phase_train:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test

        label = np.load(label_dir[idx])
        depth = imageio.imread(depth_dir[idx])
        image = imageio.imread(img_dir[idx])

        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

# - RandomHSV: 샘플의 색상 공간을 무작위로 조정합니다.
class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}

# - scaleNorm: 이미지와 깊이, 레이블을 지정된 높이와 너비로 조정합니다.
class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}

# RandomScale: 무작위로 이미지의 크기를 조정합니다.
class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}

# RandomCrop: 이미지에서 무작위로 특정 크기의 부분을 잘라냅니다.
class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


# RandomFlip: 이미지를 무작위로 좌우 반전시킵니다.
class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
# Normalize: 이미지와 깊이 데이터를 정규화합니다.
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        depth = torchvision.transforms.Normalize(mean=[19050],
                                                 std=[9650])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample

# ToTensor: 샘플 데이터를 PyTorch 텐서로 변환합니다.
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Generate different label scales
        label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                                          order=0, mode='reflect', preserve_range=True)
        label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                          order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)

        # numpy 배열을 PyTorch 텐서로 변환
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),
                'label2': torch.from_numpy(label2).float(),
                'label3': torch.from_numpy(label3).float(),
                'label4': torch.from_numpy(label4).float(),
                'label5': torch.from_numpy(label5).float()}
