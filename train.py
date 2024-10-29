from __future__ import print_function
import argparse
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set
from cod_pred import pred_BGNet,pred_SINet
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from  pytorch_ssim_master import pytorch_ssim
loss_f_mean = nn.CrossEntropyLoss(weight=None, reduction='mean')


# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50,
                    help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()


# print(opt)


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img



def img_show_plt(img):
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(img)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    # 必须有这个，要不然无法显示
    plt.show()


def img_show(img):
    a = img.squeeze().cpu().permute(1, 2, 0).numpy()
    cv2.imshow('a', a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_show_3(img):
    a = img.cpu().numpy()
    cv2.imshow('pred', a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_show_2(img):
    a = img.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    cv2.imshow('a', a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ssim_loss(warp_flow1, input_flow12):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(warp_flow1, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(input_flow12, kernel_size=3, stride=1, padding=1)
    sigma_x = torch.sqrt(F.avg_pool2d(warp_flow1 ** 2, kernel_size=3, stride=1, padding=1) - mu_x ** 2 + c1)
    sigma_y = torch.sqrt(F.avg_pool2d(input_flow12 ** 2, kernel_size=3, stride=1, padding=1) - mu_y ** 2 + c1)
    sigma_xy = F.avg_pool2d(warp_flow1 * input_flow12, kernel_size=3, stride=1, padding=1) - mu_x * mu_y + c2

    ssim = ((2 * mu_x * mu_y + 0.0001) * (2 * sigma_xy + 0.0009)) / (
            (mu_x ** 2 + mu_y ** 2 + 0.0001) * (sigma_x ** 2 + sigma_y ** 2 + 0.0009))

    ssim_loss = 1 - torch.mean(ssim, dim=(1, 2, 3), keepdim=True)

    diff_flow = warp_flow1 - input_flow12

    return diff_flow * ssim_loss


opt.dataset = 'facades'
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "./dataset/data/"
train_set, train_name_list = get_training_set(root_path + '', opt.direction)
test_set, test_name_list = get_test_set(root_path + '', opt.direction)
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size,
                                  shuffle=True)  # num_workers=opt.threads
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.test_batch_size, shuffle=False)

device = torch.device("cuda:0")
print(device)

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device) # L1-->MAE
criterionMSE = nn.MSELoss().to(device)  #L2 损失L2-->mean loss  Mean-Squared Error  L2-->MSE
ssim_loss = pytorch_ssim.SSIM(window_size = 11).to(device)

# setup optimizer
# for parameters in net_g.parameters():
#     # print(parameters)

optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

one=0
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    i = 0
    sum=0
    for iteration, batch in enumerate(training_data_loader, 1):
        o_img = Image.open(os.path.join(root_path + 'train/imgs/', train_name_list[i])).convert('RGB')
        o_img1 = transforms.ToTensor()(o_img).cuda()
        size1, size2, size3 = o_img1.shape

        o_img = o_img.resize((256, 256), Image.BICUBIC)
        o_img = transforms.ToTensor()(o_img).cuda()
        # A = pred(o_img.unsqueeze(0)).cuda(0)
        # img_show_3(A)
        # A=A.resize((256,256), Image.BICUBIC)

        # o_img=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(o_img)
        # img_show_plt(o_img)
        reverse_gt = Image.open(os.path.join(root_path + 'train/reverse_gt/', train_name_list[i].strip('jpg')+'png')).convert('L')
        reverse_gt = transforms.ToTensor()(reverse_gt).cuda()


        real_a, real_b = batch[0].to(device), batch[1].to(device)

        # img_show(real_a*0.5+0.5)
        # img_show(real_b)
        fake_b = net_g(real_a, 'aaa')  # ,size2,size3
        # noise,weight = net_g(real_a, 'weight')
        # weight=F.sigmoid(weight)
        # print(weight)
        # weight =F.sigmoid(weight)
        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb #就是一个输入 反馈一个结果  一个计算过程
        loss_g = loss_g_gan + loss_g_l1 # 在这里进行改动即可  完全可以忽略image to image的处理过程  它只要反馈给我一个背景纹理就好了

        fake_b = F.interpolate(fake_b, size=(size2, size3), mode='bilinear', align_corners=False)
        # img_show_2(noise)
        # I_adv = weight * fake_b + (1 - weight) * o_img1
        ToTensor_transform = transforms.Compose([transforms.ToTensor()])
        fake_b = transform_convert(fake_b.squeeze().cpu(), ToTensor_transform)
        o_img2=transform_convert(o_img1.squeeze().cpu(), ToTensor_transform)

        I_ADV = Image.blend(o_img2, fake_b, 0.17)  # alphe=0.3  img1*(1-0.3) + img2*0.3
        # print(weight)
        I_ADV = transforms.ToTensor()(I_ADV).cuda()  # 3 n n

        # img_show_2(I_adv)
        # img_show_2(I_adv)
        # Loss_ssim=1-torch.cosine_similarity(real_b, I_adv)
        # Loss_ssim=1-ms_ssim(real_b,I_adv)
        Loss_ssim = 1 - ssim_loss(o_img1.unsqueeze(0), I_ADV.unsqueeze(0))  #计算原图与ADV图之间的差距  缩小差距
        # print('L_ssim')
        # print(Loss_ssim)
        a, b, c, d = real_a.shape
        # Loss_ssim = criterionL1(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(o_img1), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(I_adv))  # 控制原图和生成的ADV图之间的差异

        black_GT = torch.zeros(size=(size2, size3)).cuda()
        # img_show_3(black_GT)
        # A = pred_BGNet(I_ADV.unsqueeze(0)).cuda(0)  #1 3 n n\
        A = pred_SINet(I_ADV.unsqueeze(0)).cuda(0)  # 1 3 n n\
        # img_show_3(A)

        # M = MAE()
        # loss_adv= M.step(pred=pred, gt=black_GT)
        # loss_adv=1-torch.cosine_similarity(A,black_GT)
        # loss_adv = 1 - ms_ssim(A.unsqueeze(0).unsqueeze(0),black_GT.unsqueeze(0))
        # loss_mean = ms_ssim(A,black_GT)
        loss_mean = criterionL1(black_GT, A.squeeze())  # 控制预测结果  计算预测结果和等size的纯黑图之间的损失
        # reverse_gt
        # loss_mean = criterionL1(A.squeeze(), reverse_gt.squeeze())
        # loss_mean= criterionMSE(black_GT, A)
        print('L_mean')
        print(loss_mean)

        # loss_adv=criterionL1(A, black_GT)

        Loss_total = loss_g/5+loss_mean*10  #loss——g通常最后稳定到5-6之间  +Loss_ssim*10
        Loss_total.backward()
        # Loss_total.backward()
        # loss_g.backward()

        optimizer_g.step()
        sum = sum + float(loss_mean)
        i = i + 1
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), Loss_total.item()))

    print('qian', one, 'hou', sum, 'best')
    if(one > sum):
        torch.save(net_g, "./checkpoint/{}_2/netG_SINet_model_epoch_best.pth".format('test'))
        one = sum
    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)

        prediction = net_g(input, 'a')
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    # checkpoint
    if epoch % 50 == 0:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint", exist_ok=True)
        if not os.path.exists(os.path.join("./checkpoint", opt.dataset)):
            os.makedirs(os.path.join("./checkpoint", opt.dataset), exist_ok=True)
        net_g_model_out_path = "./checkpoint/{}_2/netG_SINet_model_epoch_{}.pth".format('test', epoch)
        net_d_model_out_path = "./checkpoint/{}_2/netD_SINet_model_epoch_{}.pth".format('test', epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("./checkpoint" + opt.dataset))
