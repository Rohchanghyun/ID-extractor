from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from opt import opt
import os
import re
import random
import torch


class Data():
    def __init__(self):
        train_transform = transforms.Compose([
            # 이미지를 RGB 모드로 변환하여 투명도 채널 제거
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.vis_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor()
        ])
        
        # 데이터셋 경로 설정
        train_dir = os.path.join(opt.data_path, 'train_limit')
        test_dir = os.path.join(opt.data_path, 'test')
        query_dir = os.path.join(opt.data_path, 'query')

        # Train IDs 가져오기
        train_ids = [
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
        train_ids.sort()  # 일관된 순서를 위해 정렬

        # Test IDs 가져오기 (test와 query는 동일한 ID를 가집니다)
        test_ids = [
            d for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d))
        ]
        test_ids.sort()

        # 테스트 및 쿼리 세트를 위한 id2label 매핑 생성
        id2label = {id_name: idx for idx, id_name in enumerate(test_ids)}

        # 데이터셋 생성
        self.trainset = Top15Dataset(train_transform, 'train', train_dir)
        self.testset = Top15Dataset(test_transform, 'test', test_dir, id2label=id2label)
        self.queryset = Top15Dataset(test_transform, 'query', query_dir, id2label=id2label)

        # 데이터로더 생성
        self.train_loader = DataLoader(
            self.trainset,
            sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                  batch_image=opt.batchimage),
            batch_size=opt.batchid * opt.batchimage,
            num_workers=8,
            pin_memory=True
        )
        self.test_loader = DataLoader(self.testset, batch_size=opt.batchtest,
                                      num_workers=8, pin_memory=True)
        self.query_loader = DataLoader(self.queryset, batch_size=opt.batchtest,
                                       num_workers=8, pin_memory=True)

        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))


class Top15Dataset(Dataset):
    def __init__(self, transform, dtype, data_path, id2label=None):
        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path
        self.dtype = dtype

        # 데이터 경로에서 모든 ID 디렉토리 가져오기
        self.id_dirs = [
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ]
        self.id_dirs.sort()  # 일관된 순서를 위해 정렬

        # id2label 매핑 생성 또는 기존 매핑 사용
        if id2label is None:
            self.id2label = {id_name: idx for idx, id_name in enumerate(self.id_dirs)}
        else:
            self.id2label = id2label

        # 이미지와 라벨 수집
        self.imgs = []
        self.labels = []
        for id_name in self.id_dirs:
            id_dir = os.path.join(self.data_path, id_name)
            label = self.id2label[id_name]
            image_files = self.list_pictures(id_dir)
            for img_path in image_files:
                self.imgs.append(img_path)
                self.labels.append(label)

        # IDs와 카메라 IDs 할당
        self._ids = [self.id(path) for path in self.imgs]
        if dtype == 'query':
            self._cameras = [0] * len(self.imgs)  # 쿼리 이미지에 카메라 ID 0 할당
        else:
            self._cameras = [1] * len(self.imgs)  # 테스트/갤러리 이미지에 카메라 ID 1 할당

        self._unique_ids = sorted(set(self._ids))

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]

        img = self.loader(path)
        if img.mode == 'P':
            img = img.convert('RGBA')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: 파일 경로 (유닉스 스타일)
        :return: 사람 ID
        """
        # 파일 경로에서 ID 폴더 이름을 추출합니다.
        return os.path.basename(os.path.dirname(file_path))

    @staticmethod
    def camera(file_path):
        """
        :param file_path: 파일 경로 (유닉스 스타일)
        :return: 카메라 ID
        """
        # 파일 이름에서 카메라 ID를 추출합니다.
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) > 1:
            # 예: 파일명이 'img_c1_001.jpg'인 경우
            for part in parts:
                if part.startswith('c'):
                    return int(part[1:])
        return 0  # 기본값

    @property
    def ids(self):
        """
        :return: 데이터셋 이미지 경로에 해당하는 사람 ID 리스트
        """
        return self._ids

    @property
    def unique_ids(self):
        """
        :return: 오름차순으로 정렬된 고유한 사람 ID 리스트
        """
        return self._unique_ids

    @property
    def cameras(self):
        """
        :return: 데이터셋 이미지 경로에 해당하는 카메라 ID 리스트
        """
        return self._cameras

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), '데이터셋이 존재하지 않습니다: {}'.format(directory)
        return sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f, flags=re.IGNORECASE)
        ])