import argparse
import time
import torch
import numpy as np
import torch.optim as optim
import os
import random
import skimage.transform
from torchsummary import summary

# custom modules
from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader

# plot params

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)

#Seeding
new_manual_seed = 0
torch.manual_seed(new_manual_seed)
torch.cuda.manual_seed_all(new_manual_seed)
np.random.seed(new_manual_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(new_manual_seed)
os.environ['PYTHONHASHSEED'] = str(new_manual_seed)


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('--train_dir', default='/media/akosta/Datasets/KITTI_processed/train/',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images'
                        )
    parser.add_argument('--val_dir', default='/media/akosta/Datasets/KITTI_processed/val/',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images'
                        )
    parser.add_argument('--test_dir', default='/media/akosta/Datasets/KITTI_processed/test/',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images'
                        )
    parser.add_argument('--model_path', default=None, 
                        help='path to the trained model')

    parser.add_argument('--output_directory', default='./results/exp1/',
                        help='where save dispairities\
                        for tested images'
                        )

    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)

    parser.add_argument('--model', default='resnet18_md',
                        help='encoder architecture: ' +
                        'resnet18_md or resnet50_md ' + '(default: resnet18)'
                        + 'or torchvision version of any resnet model'
                        )

    parser.add_argument('--pretrained', default=True,
                        help='Use weights of pretrained model'
                        )

    parser.add_argument('--mode', default='train',
                        help='mode: train or test (default: train)')

    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='number of total epochs to run')

    parser.add_argument('-lr', '--learning_rate',  type=float, default=1e-4,
                        help='initial learning rate (default: 1e-4)')

    parser.add_argument('-b', '--batch_size', type=int, default=12,
                        help='mini-batch size (default: 24)')

    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )

    parser.add_argument('--device', default='cuda',
                        help='choose cpu or cuda device"'
                        )

    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')

    parser.add_argument('--augment_parameters', default=[
        0.8,
        1.2,
        0.5,
        2.0,
        0.8,
        1.2,
        ],
            help='lowest and highest values for gamma,\
                        brightness and color respectively'
            )

    parser.add_argument('--print_images', default=False,
                        help='print disparity and image\
                        generated from disparity on every iteration'
                        )
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')

    parser.add_argument('-j', '--num_workers',  type=int, default=8,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=True)
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU device ids to use')
    parser.add_argument('-t', '--threads', type=int, default=32,
                        help='Number of threads to use') 

    parser.add_argument('--save-images', action='store_true', default=False,
                        help='save disparity and image\
                        generated from disparity on every iteration'
                        )

    args = parser.parse_args()
    return args

def print_args(args):
    print(' ' * 10 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 10 + k + ': ' + str(v))

def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""

    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class Model:

    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        self.model = get_model(args.model, input_channels=args.input_channels, pretrained=args.pretrained)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        if not os.path.exists(self.args.output_directory):
            os.makedirs(self.args.output_directory)

        if args.mode == 'train':
            self.loss_function = MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate)
            self.train_n_img, self.train_loader = prepare_dataloader(args.train_dir, args.mode,
                                                                 args.augment_parameters,
                                                                 True, args.batch_size,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers)

            self.val_n_img, self.val_loader = prepare_dataloader(args.val_dir, args.mode,
                                                                 args.augment_parameters,
                                                                 False, args.batch_size,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers)
        else:
            if args.model_path:
                self.model.load_state_dict(torch.load(args.model_path))
                args.augment_parameters = None
                args.do_augmentation = False
                args.batch_size = 1
            else:
                raise Exception('Pretrained model path not provided!')

        self.test_n_img, self.test_loader = prepare_dataloader(args.test_dir, 'test', None,
                                                    False, args.batch_size,
                                                    (args.input_height, args.input_width),
                                                    args.num_workers)

        # Load data
        self.output_directory = args.output_directory
        self.input_height = args.input_height
        self.input_width = args.input_width

        if 'cuda' in self.device:
            torch.cuda.synchronize()


    def train(self):
        losses = []
        val_losses = []
        best_loss = float('Inf')
        best_val_loss = float('Inf')

        running_val_loss = 0.0
        self.model.eval()
        for data in self.val_loader:
            data = to_device(data, self.device)
            left = data['left_image']
            right = data['right_image']
            disps = self.model(left)
            loss = self.loss_function(disps, [left, right])
            val_losses.append(loss.item())
            running_val_loss += loss.item()

        running_val_loss /= self.val_n_img / self.args.batch_size
        print('Val_loss:', running_val_loss)

        for epoch in range(self.args.epochs):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.model.train()
            for data in self.train_loader:
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']

                # One optimization iteration
                self.optimizer.zero_grad()
                disps = self.model(left)
                loss = self.loss_function(disps, [left, right])
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                # Print statistics
                if self.args.print_weights:
                    j = 1
                    for (name, parameter) in self.model.named_parameters():
                        if name.split(sep='.')[-1] == 'weight':
                            plt.subplot(5, 9, j)
                            plt.hist(parameter.data.view(-1))
                            plt.xlim([-1, 1])
                            plt.title(name.split(sep='.')[0])
                            j += 1
                    plt.show()

                if self.args.print_images:
                    print('disp_left_est[0]')
                    plt.imshow(np.squeeze(
                        np.transpose(self.loss_function.disp_left_est[0][0,
                                     :, :, :].cpu().detach().numpy(),
                                     (1, 2, 0))))
                    plt.show()
                    print('left_est[0]')
                    plt.imshow(np.transpose(self.loss_function\
                        .left_est[0][0, :, :, :].cpu().detach().numpy(),
                        (1, 2, 0)))
                    plt.show()
                    print('disp_right_est[0]')
                    plt.imshow(np.squeeze(
                        np.transpose(self.loss_function.disp_right_est[0][0,
                                     :, :, :].cpu().detach().numpy(),
                                     (1, 2, 0))))
                    plt.show()
                    print('right_est[0]')
                    plt.imshow(np.transpose(self.loss_function.right_est[0][0,
                               :, :, :].cpu().detach().numpy(), (1, 2,
                               0)))
                    plt.show()
                running_loss += loss.item()

            running_val_loss = 0.0
            self.model.eval()
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                disps = self.model(left)
                loss = self.loss_function(disps, [left, right])
                val_losses.append(loss.item())
                running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.train_n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print (
                '\nEpoch: [{}/{}] \t Train_loss: {:10.4f} \t Val_loss: {:10.4f} \t Time: {:10.3f}secs'.format(epoch + 1,
                self.args.epochs,
                running_loss, 
                running_val_loss, 
                time.time() - c_time))
 
            self.save(os.path.join(self.output_directory, 'model_' + self.args.model +  '_cpt.pth'))

            if running_loss < best_loss:
                self.save(os.path.join(self.output_directory, 'model_' + self.args.model +  '_best.pth'))
                best_loss = running_loss
                print('Best_train Model_saved')

            if running_val_loss < best_val_loss:
                self.save(os.path.join(self.output_directory, 'model_' + self.args.model +  '_best_val.pth'))
                best_val_loss = running_val_loss
                print('Best_val Model_saved')

        print ('Finished Training! \nBest loss: ', best_loss)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def test(self):
        self.model.eval()
        disparities = np.zeros((self.test_n_img,
                               self.input_height, self.input_width),
                               dtype=np.float32)
        disparities_pp = np.zeros((self.test_n_img,
                                  self.input_height, self.input_width),
                                  dtype=np.float32)
        with torch.no_grad():
            for (i, data) in enumerate(self.test_loader):
                # Get the inputs
                data = to_device(data, self.device)
                left = data.squeeze()
                # Do a forward pass
                disps = self.model(left)
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
                disparities_pp[i] = \
                    post_process_disparity(disps[0][:, 0, :, :]\
                                           .cpu().numpy())

        np.save(self.output_directory + '/disparities.npy', disparities)
        np.save(self.output_directory + '/disparities_pp.npy',
                disparities_pp)
        print('Finished Testing')


def main():
    args = return_arguments()
    print_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.set_num_threads(args.threads)

    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        print(model_test.model)
        summary(model_test.model, (3, 256, 512))
        model_test.test()
        if args.save_images:
            img_dir = os.path.join(args.output_directory, 'images')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            disp = np.load(os.path.join(args.output_directory, 'disparities_pp.npy'))  # Or disparities.npy for output without post-processing
            print('Disparity Map shape: ', disp.shape)

            for i in range(disp.shape[0]):
                disp_to_img = skimage.transform.resize(disp[i].squeeze(), [375, 1242], mode='constant')
                plt.imsave(os.path.join(img_dir, 'pred_'+str(i)+'.png'), disp_to_img, cmap='plasma')

            print('Disparity map images saved.')


if __name__ == '__main__':
    main()

