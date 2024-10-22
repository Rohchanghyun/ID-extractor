from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from opt import opt
import os
import re
import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib
import pdb
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from opt import opt
from data import Data
from network import MGN
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_single_feature
from utils.metrics import mean_ap, cmc, re_ranking
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import os
class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.loader = default_loader
        vis_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        img = self.loader(self.image_path)
        original_size = img.size  # 원래 이미지 크기를 저장
        if self.transform is not None:
            img = self.transform(img)
        return img, original_size
    def __len__(self):
        return 1
    

class Main():
    def __init__(self, model, loss, data):
        self.model = model.to('cuda')
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)
    def evaluate(self):
        self.model.eval()
        # 예제 이미지 경로
        image_path = '../dataset/top15character/query/Toge Inumaki/888.png'
        
        # 트랜스폼 정의
        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 데이터셋 및 데이터로더 생성
        single_image_dataset = SingleImageDataset(image_path, transform=test_transform)
        single_image_loader = DataLoader(single_image_dataset, batch_size=1, num_workers=0, pin_memory=True)
        # 데이터로더 사용 예제
        for img, original_size in single_image_loader:
            
            gf, size, p1, p2, p3 = extract_single_feature(self.model, tqdm(single_image_loader))
            gf = gf.numpy()
            size = original_size
            break
        
        # attention
        self.attention([p1,p2,p3],image_path,size)
        
        
    def attention(self, part, image_path, size):
        for i, branch in enumerate(part):
            # 파일 경로에서 이미지 ID 추출
            id = os.path.basename(os.path.dirname(image_path))
            image_name = os.path.basename(image_path).split('.')[0]
            # feature map을 원본 이미지 크기로 upsampling
            upsampled_feature = F.interpolate(branch, size=tuple(size[::-1]), mode='bicubic', align_corners=False)
            
            # feature map 정규화
            norm_feature = torch.norm(upsampled_feature, dim=1)
            
            # 원본 이미지 로드 및 데이터 로더 설정
            single_image_dataset = SingleImageDataset(image_path, transform=data.vis_transform)
            single_image_loader = DataLoader(single_image_dataset, batch_size=1, num_workers=0, pin_memory=True)
            
            # 데이터 로더에서 이미지 하나 로드
            for img, original_size in single_image_loader:
                original_img = img
                break
            
            # feature map과 원본 이미지 텐서를 CPU로 옮기고 시각화 준비
            final_result = norm_feature.squeeze().detach().cpu()
            final_result = torch.flip(final_result, dims=[1])
            original_image = original_img.squeeze().detach().cpu()

            # 시각화 - 원본 이미지와 feature map overlay
            plt.subplot(1, 3, i + 1)  # 1행 3열의 서브플롯 중 i+1번째
            plt.imshow(original_image.permute(1, 2, 0))  # (C, H, W) -> (H, W, C)로 변환
            plt.imshow(final_result, cmap='plasma', alpha=0.7)  # feature map 덧씌우기
            plt.title(f'part {i+1}')
            plt.axis('off')
            
        save_dir = "./results/vis/256_limit"
        os.makedirs(save_dir, exist_ok=True)
        # 최종 이미지 저장
        plt.savefig(f"{save_dir}/attention_{id}_{image_name}.png")
        
if __name__ == '__main__':
    data = Data()
    model = MGN()
    loss = Loss()
    main = Main(model, loss, data)
    print('start evaluate')
    model.load_state_dict(torch.load(opt.weight))
    main.evaluate()