import torch
from torch.optim import Adam, SGD
from torch.autograd import Variable
from networks.poolnet import build_model, weights_init
import numpy as np
import os
import cv2
import time
from torchvision.utils import make_grid
import torch.nn.functional as F

# modified by sierkinhane
class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [15,]
        self.build_model()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model))
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)
        if self.config.cuda:
            self.net = self.net.cuda()
        # self.net.train()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)

        if self.config.load == '':
            self.net.base.load_pretrained_model(torch.load(self.config.pretrained_model))
        else:
            self.net.load_state_dict(torch.load(self.config.load))

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

    def test(self):
        mode_name = 'sal_fuse'
        t0 = time.time()
        total = len(self.test_loader)
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                x, image_id, h, w = (
                    batch['sal_image'], batch['image_id'][0],
                    batch["height"][0], batch["width"][0]
                )
                image_path = os.path.join(self.config.sal_folder, image_id + '.png')
                if os.path.isfile(image_path):
                    continue

                if h <= 100 or w <= 100:
                    x = F.interpolate(x, scale_factor=2, mode='bilinear') # 71 --> 142

                x = Variable(x)
                if self.config.cuda:
                    x = x.cuda()
                preds = self.net(x)

                if h <= 100 or w <= 100:
                    preds = F.interpolate(preds, size=(h, w), mode='bilinear')

                preds = preds[..., :h, :w]
                pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                cv2.imwrite(image_path, 255 * pred)

                if (step + 1) % (total // 10) == 0:
                    print(f"step={step}/{total} fps={step / (time.time()-t0):.1f}")


        time_e = time.time()
        print('Speed: %f FPS' % (total/(time_e-t0)))
        print('Test Done!')

    # training phase
    def train(self):
        mode_name = 'sal_fuse'
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        for epoch in range(self.config.epoch):
            r_sal_loss= 0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label, image_id = data_batch['sal_image'], data_batch['sal_label'], data_batch['image_id'][0]
                # print(sal_image.shape, sal_label.shape)
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                b, c, h, w = sal_image.shape
                if h <= 100 or w <= 100:
                    continue
                sal_image, sal_label= Variable(sal_image), Variable(sal_label)
                if self.config.cuda:
                    # cudnn.benchmark = True
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

                sal_pred = self.net(sal_image)
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data

                sal_loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.show_every // self.config.batch_size) == 0:
                    if i == 0:
                        x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num, r_sal_loss/x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_sal_loss= 0
                if i % (4*self.show_every // self.config.batch_size) == 0:
                    pred = np.squeeze(torch.sigmoid(sal_pred).cpu().data.numpy())
                    multi_fuse = 255 * pred
                    cv2.imwrite(os.path.join(self.config.test_fold, image_id + '_sal' + '.png'), multi_fuse)

                    grid = make_grid(sal_image, nrow=1, padding=0, pad_value=0,
                         normalize=True, value_range=None)
                    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
                    image = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    cv2.imwrite(os.path.join(self.config.test_fold, image_id + '.png'), image)


            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)

def bce2d(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)
