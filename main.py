import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib
from PIL import Image
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import torch
from torch.optim import lr_scheduler

from opt import opt
from data import Data
from network import MGN
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to('cuda')
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):

        self.scheduler.step()

        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            pdb.set_trace()
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def evaluate(self):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

    def vis(self):
        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('extract features, this may take a few minutes')
        query_feature = extract_feature(self.model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(self.model, tqdm(data.test_loader))
        
        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        # Show query image
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')
        
        # Query image의 폴더 ID 가져오기
        query_folder = os.path.basename(os.path.dirname(opt.query_image))

        for i in range(10):
            img_path = gallery_path[index[i]]
            print(img_path)

            # Gallery image의 폴더 ID 가져오기
            gallery_folder = os.path.basename(os.path.dirname(img_path))

            # 이미지 로드 및 표시
            ax = plt.subplot(1, 11, i + 2)
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')

            # 같은 폴더 ID이면 초록색, 아니면 빨간색 테두리 추가
            if query_folder == gallery_folder:
                color = 'green'
            else:
                color = 'red'

            # 테두리 그리기
            rect = patches.Rectangle((0, 0), img.width, img.height, linewidth=5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig("show.png")
        print('result saved to show.png')


if __name__ == '__main__':

    data = Data()
    model = MGN()
    loss = Loss()
    main = Main(model, loss, data)

    if hasattr(opt, 'weight') and opt.weight:
        print(f"Loading pretrained weights from {opt.weight}")
        model.load_state_dict(torch.load(opt.weight))
    if opt.mode == 'train':

        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            main.train()
            if epoch % 1000 == 0:
                print('\nstart evaluate')
                #main.evaluate()
                os.makedirs('weights/256_limit_resume', exist_ok=True)
                torch.save(model.state_dict(), ('weights/256_limit_resume/model_{}.pt'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()

    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.weight))
        main.vis()
