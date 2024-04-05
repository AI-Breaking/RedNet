import torch
from torch import nn
import math
import torch.utils.model_zoo as model_zoo
from utils import utils
from torch.utils.checkpoint import checkpoint


class RedNet(nn.Module):
    def __init__(self, num_classes=37, pretrained=False):

        super(RedNet, self).__init__()
        block = Bottleneck
        transblock = TransBasicBlock
        layers = [3, 4, 6, 3]
        # original resnet
        self.inplanes = 64

        # 입력 이미지에 대해 2D 합성곱 연산을 수행합니다. 
        # 이 연산은 7x7 크기의 필터를 사용하여 입력 이미지의 각 3채널(RGB)에서 특징을 추출하고, 64개의 다른 필터를 적용하여 64개의 출력 채널을 생성합니다. 
        # stride=2는 필터가 이미지 위를 이동할 때 두 픽셀씩 건너뛴다는 것을 의미하며, 
        # padding=3은 이미지의 가장자리에 3픽셀의 패딩을 추가하여 모서리 주변의 정보도 잘 포착할 수 있도록 합니다.
        # 가중치 형태: (64, 3, 7, 7)에서, 64는 출력 채널의 수, 3은 입력 채널의 수, 7x7은 커널(필터)의 크기입니다.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # nn.BatchNorm2d(64)는 합성곱 연산을 통해 생성된 64개의 출력 채널 각각에 대해 배치 정규화를 수행합니다. 
        # 이는 학습 과정을 안정화하고 가속화하는 데 도움이 됩니다. 
        self.bn1 = nn.BatchNorm2d(64)
        # nn.ReLU(inplace=True)는 비선형 활성화 함수로, 음수 값을 모두 0으로 설정하여 네트워크의 비선형성을 증가시킵니다.
        self.relu = nn.ReLU(inplace=True)
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)는 최대 풀링 연산을 수행하여, 3x3 영역 내에서 가장 큰 값을 선택합니다. 
        # stride=2는 이 영역이 두 픽셀씩 건너뛰며 이동한다는 것을 의미합니다. 이는 특징 맵의 크기를 줄이면서 주요 특징을 유지하는 데 도움이 됩니다.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 연산: self._make_layer(block, 64, layers[0])와 같은 라인은, 지정된 block 타입(Bottleneck)을 사용하여 여러 레이어를 생성합니다. 
        # layers 배열은 각 ResNet 블록에서 몇 개의 Bottleneck 레이어를 쌓을지 결정합니다. 
        # 이 구조를 통해 모델은 점점 더 추상적이고 고수준의 특징을 추출할 수 있게 됩니다.
        # 가중치 형태: Bottleneck 내부의 합성곱 레이어들은 다양한 크기의 가중치를 가집니다. 
        # 예를 들어, 첫 번째 1x1 합성곱은 차원 축소를 위해 사용되며, 두 번째 3x3 합성곱은 중요한 특징을 추출하고, 마지막 1x1 합성곱은 차원을 확장합니다.
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # resnet for depth channel
        self.inplanes = 64
        # conv1_d는 (64, 1, 7, 7)로, 입력 채널이 깊이 정보 하나이기 때문에 1입니다.
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        # 연산: 이 레이어들은 TransBasicBlock 블록을 사용해 구성되며, 전치 합성곱(transposed convolution, 또는 deconvolution)을 통해 업샘플링을 수행합니다. 
        # 전치 합성곱은 합성곱의 반대 과정으로, 특징 맵의 크기를 확장합니다. stride=2 설정은 특징 맵의 크기를 2배로 증가시킵니다.
        # 가중치 형태: 전치 합성곱 레이어의 가중치 형태는 (입력 채널 수, 출력 채널 수, 커널 크기, 커널 크기)로, 
        # 예를 들어 deconv1의 가중치 형태는 (256, 512, kernel_size, kernel_size)가 될 수 있습니다(여기서 kernel_size는 _make_transpose 함수 내에서 설정됩니다).
        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        # 연산: 이 레이어들은 1x1 합성곱을 사용하여 특징 맵의 채널 수를 조정합니다. 
        # 이 과정은 업샘플링된 특징 맵과 이전 단계의 특징 맵을 효율적으로 결합하기 위해 사용됩니다. 
        # 이를 통해 네트워크는 다양한 해상도에서 추출된 특징들을 종합하여 더 정확한 예측을 할 수 있습니다.
        # 가중치 형태: (출력 채널 수, 입력 채널 수, 1, 1)로, 예를 들어 agant0는 (64, 64, 1, 1)의 가중치를 가집니다. 
        # 여기서 1x1 합성곱은 주로 채널 수를 변환하는 데 사용되며, 공간적 차원은 변화시키지 않습니다.
        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64 * 4, 64)
        self.agant2 = self._make_agant_layer(128 * 4, 128)
        self.agant3 = self._make_agant_layer(256 * 4, 256)
        self.agant4 = self._make_agant_layer(512 * 4, 512)

        # final block
        self.inplanes = 64
        # final_conv: _make_transpose 함수를 통해 정의되며, TransBasicBlock을 사용하여 업샘플링을 수행합니다. 
        # 이 과정은 최종 출력 크기에 도달하기 전, 특징 맵의 해상도를 점진적으로 높입니다.
        self.final_conv = self._make_transpose(transblock, 64, 3)
        # final_deconv: nn.ConvTranspose2d(self.inplanes, num_classes, kernel_size=2, stride=2, padding=0, bias=True)를 사용하여 최종 업샘플링을 수행합니다. 
        # 이 레이어는 모델의 최종 출력을 생성하며, 각 픽셀의 클래스 예측에 해당하는 예측 맵을 만듭니다. 
        # num_classes는 출력 채널의 수로, 모델이 분류해야 하는 클래스의 총 수입니다.
        self.final_deconv = nn.ConvTranspose2d(self.inplanes, num_classes, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        # 이 레이어들은 다양한 깊이에서의 특징 맵을 클래스 예측으로 변환합니다. 
        # 각 레이어는 모델의 중간 단계에서 생성된 특징 맵을 입력으로 받아, 해당 해상도에서의 예측을 생성합니다. 
        # 이는 모델이 학습하는 동안 중간 목표를 제공하며, 세밀한 특징을 포착하는 데 도움을 줄 수 있습니다.
        self.out5_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.out4_conv = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, bias=True)
        self.out3_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)

        # for m in self.modules(): 
        # 루프는 모든 합성곱 및 배치 정규화 레이어의 가중치를 적절히 초기화합니다. 이 초기화 과정은 모델 학습의 안정성과 수렴 속도에 중요합니다.

        # if pretrained: 
        # self._load_resnet_pretrained()는 선택적으로 사전 훈련된 ResNet 가중치를 모델에 로드하여,
        # 이미지넷 같은 대규모 데이터셋에서 학습된 유용한 특징을 활용할 수 있게 합니다.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self._load_resnet_pretrained()

    # _make_layer 메서드는 네트워크 내에서 반복되는 블록 구조를 생성하는 함수입니다. 
    # 이 메서드는 ResNet 아키텍처의 핵심 요소 중 하나로, 여러 개의 동일한 블록(예: Bottleneck 또는 BasicBlock)을 쌓아서 하나의 "레이어"를 형성합니다.
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 다운샘플링을 위한 준비: 첫 번째 블록에서 입력 특징 맵의 크기를 조절할 필요가 있는지 확인합니다. 
        # 만약 stride가 1이 아니거나 현재 inplanes(입력 채널 수)가 블록의 출력 채널 수(planes * block.expansion)와 다르다면, 
        # 다운샘플링이 필요한 것으로 판단하고 downsample 레이어를 생성합니다. 
        # 이 레이어는 1x1 합성곱과 배치 정규화로 구성되며, 특징 맵의 크기와 깊이를 조정합니다.
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        # 레이어 내 블록의 쌓기: 첫 번째 블록을 생성하고, 필요한 경우 downsample을 적용합니다. 
        # 그 후, 지정된 개수(blocks)만큼 블록을 순차적으로 쌓습니다. 
        # 첫 번째 블록을 제외한 나머지 블록들은 stride=1과 동일한 출력 채널 수(planes)를 사용합니다.
        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        # nn.Sequential로 레이어 반환: 생성된 블록들을 nn.Sequential 컨테이너에 담아 이 메서드의 결과로 반환합니다. 
        # 이렇게 함으로써, 정의된 블록들을 순차적인 모듈로써 네트워크에 쉽게 통합할 수 있습니다.
        return nn.Sequential(*layers)

    # _make_transpose 메서드는 업샘플링을 수행하는 레이어를 구성하는 함수입니다. 
    # 이 메서드는 주로 모델의 출력 부분에서 특징 맵의 해상도를 늘리는 데 사용됩니다. 
    # 업샘플링은 이미지 세그멘테이션과 같이 출력 이미지의 해상도가 입력 이미지와 같아야 하는 작업에 필수적입니다.
    def _make_transpose(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    # 가중치 모양: 이 레이어들은 주로 1x1 합성곱을 사용하여 차원을 조절합니다. 가중치의 모양은 (planes, inplanes, 1, 1)입니다.
    # 연산: 이 레이어들은 특징 맵의 채널 수를 조절하여, 업샘플링 과정에서 다른 레이어의 출력과 결합하기 쉽게 만듭니다.
    def _make_agant_layer(self, inplanes, planes):

        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url(utils.model_urls['resnet50'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith('conv1'):  # the first conv_op
                    model_dict[k] = v
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6] + '_d' + k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward_downsample(self, rgb, depth):

        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu(depth)

        fuse0 = x + depth

        x = self.maxpool(fuse0)
        depth = self.maxpool(depth)

        # block 1
        x = self.layer1(x)
        depth = self.layer1_d(depth)
        fuse1 = x + depth
        # block 2
        x = self.layer2(fuse1)
        depth = self.layer2_d(depth)
        fuse2 = x + depth
        # block 3
        x = self.layer3(fuse2)
        depth = self.layer3_d(depth)
        fuse3 = x + depth
        # block 4
        x = self.layer4(fuse3)
        depth = self.layer4_d(depth)
        fuse4 = x + depth

        return fuse0, fuse1, fuse2, fuse3, fuse4

    def forward_upsample(self, fuse0, fuse1, fuse2, fuse3, fuse4):

        agant4 = self.agant4(fuse4)
        # upsample 1
        x = self.deconv1(agant4)
        if self.training:
            out5 = self.out5_conv(x)
        x = x + self.agant3(fuse3)
        # upsample 2
        x = self.deconv2(x)
        if self.training:
            out4 = self.out4_conv(x)
        x = x + self.agant2(fuse2)
        # upsample 3
        x = self.deconv3(x)
        if self.training:
            out3 = self.out3_conv(x)
        x = x + self.agant1(fuse1)
        # upsample 4
        x = self.deconv4(x)
        if self.training:
            out2 = self.out2_conv(x)
        x = x + self.agant0(fuse0)
        # final
        x = self.final_conv(x)
        out = self.final_deconv(x)

        if self.training:
            return out, out2, out3, out4, out5

        return out

    def forward(self, rgb, depth, phase_checkpoint=False):

        if phase_checkpoint:
            depth.requires_grad_()
            fuses = checkpoint(self.forward_downsample, rgb, depth)
            out = checkpoint(self.forward_upsample, *fuses)
        else:
            fuses = self.forward_downsample(rgb, depth)
            out = self.forward_upsample(*fuses)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    # 가중치 모양: Bottleneck 블록 내의 conv1, conv2, conv3 레이어들은 각각 다른 크기의 가중치를 가집니다. 
    # 예를 들어, conv1은 차원 축소를 위해 (planes, inplanes, 1, 1), conv2는 중간 합성곱을 위해 (planes, planes, 3, 3), 
    # conv3는 차원 확장을 위해 (planes * 4, planes, 1, 1)의 가중치를 가집니다.
    # 연산: 이 블록은 먼저 차원을 축소하고, 중간 크기의 커널로 중요한 특징을 추출한 다음, 출력 차원을 확장합니다. 
    # 이 과정은 계산 효율성을 개선하면서 중요한 정보를 유지합니다.
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TransBasicBlock(nn.Module):
    expansion = 1
    # 가중치 모양: TransBasicBlock 내의 conv1은 (inplanes, inplanes, 3, 3), conv2 (전치 합성곱을 사용할 경우)는 (planes, inplanes, 3, 3)입니다. 
    # 여기서 전치 합성곱은 출력 특징 맵의 크기를 증가시키는데 사용됩니다.
    # 연산: 업샘플링은 다운샘플링 과정에서 잃어버린 공간적 정보의 일부를 복원합니다. 
    # 이는 특징 맵의 크기를 늘리는 전치 합성곱 연산을 포함합니다.
    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out
