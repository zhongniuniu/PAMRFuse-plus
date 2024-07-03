import sys

sys.path.append("..")

import visdom
import pathlib
import warnings
import logging.config
import argparse, os

import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms

from tqdm import tqdm
import torch.nn.functional
from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform
from dataloader.joint_data import JointTrainData
from models.deformable_net import DeformableNet
from models.fusion_net import FusionNet
from loss.reg_losses import LossFunction_Dense
from loss.fusion_loss import FusionLoss
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def hyper_args():
# Training settings
    parser = argparse.ArgumentParser(description="PyTorch Corss-modality Registration")
    # dataset
    parser.add_argument('--ir', default='..', type=pathlib.Path) #MRI
    parser.add_argument('--vi', default='..', type=pathlib.Path) #PAT
    parser.add_argument('--it', default='..', type=pathlib.Path) #fake MRI(PAT->MRI)
    # parser.add_argument('--MRI_map', default='..', type=pathlib.Path) #meiyouyongdao
    # parser.add_argument('--PAT_map', default='..', type=pathlib.Path) #meiyouyongdao
    # train loss weights
    parser.add_argument('--alpha', default=1.0, type=float) #meiyouyongdao
    parser.add_argument('--beta', default=20.0, type=float) #meiyouyongdao
    parser.add_argument('--theta', default=5.0, type=float) #meiyouyongdao
    # implement details
    parser.add_argument('--dim', default=64, type=int, help='AFuse feather dim')
    parser.add_argument("--batchsize", type=int, default=8, help="training batch size")
    parser.add_argument("--nEpochs", type=int, default=120, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=1e-4")
    parser.add_argument("--step", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--interval', default=20, help='record interval')
    # checkpoint
    parser.add_argument('--load_model_reg', type=str, default='..',
                        help="Location from which any pre-trained model needs to be loaded.")
    parser.add_argument('--load_model_fuse', type=str, default='..',
                        help="Location from which any pre-trained model needs to be loaded.")
    # save path of model
    parser.add_argument("--ckpt", default="/mnt/dfc_data2/project/linyusen/project/12_niuniu/niuniu/UMF-CMGR-main/cache/Joint_reg_fusion", type=str, help="path to pretrained model (default: none)")

    args = parser.parse_args()
    return args


def main(args, visdom):
    coeff_mse_loss_VF = 1.  # alpha1
    coeff_mse_loss_IF = 1.
    coeff_decomp = 2.  # alpha2 and alpha4
    coeff_tv = 5.

    clip_grad_norm_value = 0.01
    optim_step = 20
    optim_gamma = 0.5

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    log = logging.getLogger()

    epoch    = args.nEpochs
    interval = args.interval

    print("===> Creating Save Path of Checkpoints")
    cache = pathlib.Path(args.ckpt)
    if not os.path.exists(cache):
        os.makedirs(cache)

    print("===> Loading datasets")
    crop = torchvision.transforms.RandomResizedCrop(256)   #suijicaijian, yiquxiaogaigongneng
    data = JointTrainData(args.ir, args.it, args.vi, args.ir_map, args.vi_map, crop)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)

    print("===> Building models")
    RegNet = DeformableNet().to(device)
    # FuseNet = FusionNet(nfeats=args.dim).to(device)

    DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    print("===> Defining Loss fuctions")
    criterion_reg = LossFunction_Dense().to(device)
    # criterion_fus = FusionLoss(args.alpha, args.beta, args.theta).to(device)   # fusion loss quanzhong

    criteria_fusion = Fusionloss()
    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    Loss_ssim = kornia.losses.SSIM(11)

    print("===> Setting Optimizers")
    optimizer_reg = torch.optim.Adam(params=RegNet.parameters(), lr=args.lr)
    # optimizer_fus = torch.optim.Adam(params=FuseNet.parameters(), lr=args.lr)

    optimizer1 = torch.optim.Adam(
        DIDF_Encoder.parameters(), lr=1e-4, weight_decay=0)
    optimizer2 = torch.optim.Adam(
        DIDF_Decoder.parameters(), lr=1e-4, weight_decay=0)
    optimizer3 = torch.optim.Adam(
        BaseFuseLayer.parameters(), lr=1e-4, weight_decay=0)
    optimizer4 = torch.optim.Adam(
        DetailFuseLayer.parameters(), lr=1e-4, weight_decay=0)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.5) #tiaozhengxuexilv
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.5)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=20, gamma=0.5)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=20, gamma=0.5)

    print("===> Building deformation")
    affine = AffineTransform(translate=0.01)  #tianjiasuijibianhuan
    elastic = ElasticTransform(kernel_size=101, sigma=16) #tianjiasuijibianhuan

    # # TODO: optionally copy weights from a checkpoint zairuquanzhong
    # if args.load_model_reg is not None:
    #     print('Loading pre-trained RegNet checkpoint %s' % args.load_model_reg)
    #     log.info(f'Loading pre-trained checkpoint {str(args.load_model_reg)}')
    #     state = torch.load(str(args.load_model_reg))
    #     RegNet.load_state_dict(state)
    # else:
    #     print("=> no model found at '{}'".format(args.load_model_reg))
    #
    # if args.load_model_fuse is not None:
    #     print('Loading pre-trained FuseNet checkpoint %s' % args.load_model_fuse)
    #     log.info(f'Loading pre-trained checkpoint {str(args.load_model_fuse)}')
    #     state = torch.load(args.load_model_fuse)#['net']
    #     FuseNet.load_state_dict(state)
    # else:
    #     print("=> no model found at '{}'".format(args.load_model_fuse))

    # TODO: freeze parameter of RegNet
    for param in RegNet.parameters():
        param.requires_grad = True

    print("===> Starting Training")
    print(args.nEpochs)
    epoch_gap = 40
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        tqdm_loader = tqdm(training_data_loader, disable=True)
        # total_loss, reg_loss, fus_loss = Joint_train(args, tqdm_loader, optimizer_reg, optimizer_fus, RegNet, FuseNet, criterion_reg, criterion_fus, epoch, elastic, affine)
        #################################
        RegNet.train()

        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer_reg.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        # TODO: update learning rate of the optimizer
        lr_R = adjust_learning_rate(args, optimizer_reg, epoch - 1)
        # lr_F = adjust_learning_rate(args, optimizer_fus, epoch - 1)
        print("Epoch={}, lr_R={}".format(epoch, lr_R))

        loss_total, loss_reg, loss_fus = [], [], []
        for (ir, it, vi, ir_map, vi_map), _ in tqdm_loader:

            ir, it, vi = ir.cuda(), it.cuda(), vi.cuda()
            ir_map, vi_map = ir_map.cuda(), vi_map.cuda()

            if epoch < epoch_gap:
                # TODO: generate warped ir images shengchengbianxingtuxiang
                # ir_affine, ir_affine_disp = affine(ir)
                # ir_elastic, ir_elastic_disp = elastic(ir_affine)
                # disp_ir = ir_affine_disp + ir_elastic_disp  # cumulative disp grid [batch_size, height, weight, 2]
                ir_warp = ir  # ir_elastic

                ir_warp.detach_()
                # disp_ir.detach_()

                # TODO: train registration
                ir_pred, ir_f_warp, ir_flow, ir_int_flow1, ir_int_flow2, ir_disp_pre = RegNet(it, ir_warp, vi)
                reg_loss, vgg, ncc, grad = criterion_reg(ir_pred, ir_f_warp, it, ir_warp, ir_flow, ir_int_flow1,
                                                         ir_int_flow2)

                print("vgg={}, ncc={}, grad={}".format(vgg, ncc, grad))

                # TODO: train fusion
                feature_V_B, feature_V_D, _ = DIDF_Encoder(vi)
                feature_I_B, feature_I_D, _ = DIDF_Encoder(ir_pred)
                data_VIS_hat, _ = DIDF_Decoder(vi, feature_V_B, feature_V_D)
                data_IR_hat, _ = DIDF_Decoder(ir_pred, feature_I_B, feature_I_D)

                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                mse_loss_V = Loss_ssim(vi, data_VIS_hat) + MSELoss(vi, data_VIS_hat)
                mse_loss_I = 10 * Loss_ssim(ir_pred, data_IR_hat) + MSELoss(ir_pred, data_IR_hat)
                mse_loss_I = torch.mean(mse_loss_I)
                mse_loss_V = torch.mean(mse_loss_V)
                MSEI = mse_loss_I
                MSEV = mse_loss_V

                Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(vi),kornia.filters.SpatialGradient()(data_VIS_hat)) + 5 * L1Loss(kornia.filters.SpatialGradient()(ir_pred),kornia.filters.SpatialGradient()(data_IR_hat))

                loss_decomp = abs(cc_loss_D) + abs(cc_loss_B) #(cc_loss_D) ** 2 / (1.01 + cc_loss_B)

                fusion_loss = 15. * abs(mse_loss_V) + 15. * abs(mse_loss_I) + 10. * loss_decomp + 25. * Gradient_loss

                print("cc_loss_B={}, cc_loss_D={}, mse_loss_V={}, mse_loss_I={}, Gradient_loss={}".format(cc_loss_B, cc_loss_D, MSEI, MSEV, Gradient_loss))
                # fusion_loss = fusion_loss.mean()

                reg_loss.backward(retain_graph=True)

                fusion_loss.backward()

                nn.utils.clip_grad_norm_(
                    DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                optimizer1.step()
                optimizer2.step()
                optimizer_reg.step()

            else:  # Phase II
                # ir_affine, ir_affine_disp = affine(ir)
                # ir_elastic, ir_elastic_disp = elastic(ir_affine)
                # disp_ir = ir_affine_disp + ir_elastic_disp  # cumulative disp grid [batch_size, height, weight, 2]
                ir_warp = ir  # ir_elastic
                ir_warp.detach_()
                # disp_ir.detach_()

                ir_pred, ir_f_warp, ir_flow, ir_int_flow1, ir_int_flow2, ir_disp_pre = RegNet(it, ir_warp, vi)
                reg_loss, vgg, ncc, grad = criterion_reg(ir_pred, ir_f_warp, it, ir_warp, ir_flow, ir_int_flow1,
                                                         ir_int_flow2)

                print("vgg={}, ncc={}, grad={}".format(vgg, ncc, grad))

                feature_V_B, feature_V_D, feature_V = DIDF_Encoder(vi)
                feature_I_B, feature_I_D, feature_I = DIDF_Encoder(ir_pred)

                feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
                feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
                data_Fuse, feature_F = DIDF_Decoder(vi, feature_F_B, feature_F_D)

                mse_loss_V = Loss_ssim(vi, data_Fuse) + MSELoss(vi, data_Fuse)
                mse_loss_I = Loss_ssim(ir_pred, data_Fuse) + MSELoss(ir_pred, data_Fuse)
                mse_loss_I = torch.mean(mse_loss_I)
                mse_loss_V = torch.mean(mse_loss_V)

                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                loss_decomp = abs(cc_loss_D) + abs(cc_loss_B)
                fusionloss, _, _ = criteria_fusion(vi, ir_pred, data_Fuse)

                fusion_loss = fusionloss + 50. * loss_decomp + 100 * abs(mse_loss_I) + 10 * abs(mse_loss_V)

                print("fusionloss={}, loss_decomp={}".format(fusionloss, loss_decomp))

                reg_loss.backward(retain_graph=True)
                fusion_loss.backward()
                nn.utils.clip_grad_norm_(
                    DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()
                optimizer_reg.step()

            # batches_done = epoch * len(loader['train']) + i
            # batches_left = num_epochs * len(loader['train']) - batches_done
            # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            # prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/600][loss: %f]"
                % (
                    epoch,
                    fusion_loss.item(),
                )
            )
        scheduler1.step()
        scheduler2.step()

        if not epoch < epoch_gap:
            scheduler3.step()
            scheduler4.step()

        if optimizer1.param_groups[0]['lr'] <= 1e-6:
            optimizer1.param_groups[0]['lr'] = 1e-6
        if optimizer2.param_groups[0]['lr'] <= 1e-6:
            optimizer2.param_groups[0]['lr'] = 1e-6
        if optimizer3.param_groups[0]['lr'] <= 1e-6:
            optimizer3.param_groups[0]['lr'] = 1e-6
        if optimizer4.param_groups[0]['lr'] <= 1e-6:
            optimizer4.param_groups[0]['lr'] = 1e-6

        # TODO: total loss
        # loss = 1.0 * reg_loss + 1.0 * fusion_loss
        #
        # loss_total.append(loss.item())
        # loss_reg.append(reg_loss.item())
        # loss_fus.append(fusion_loss.item())


        checkpoint = {
            'DIDF_Encoder': DIDF_Encoder.state_dict(),
            'DIDF_Decoder': DIDF_Decoder.state_dict(),
            'BaseFuseLayer': BaseFuseLayer.state_dict(),
            'DetailFuseLayer': DetailFuseLayer.state_dict(),
        }
        torch.save(checkpoint, f'../cache/Joint_reg_fusion/fusion_{epoch:04d}.pth') if epoch % interval == 0 else None

        # l = len(loss_total)
        # total_loss = (sum(loss_total) / l)
        # reg_loss = (sum(loss_reg) / l)
        # fus_loss = (sum(loss_fus) / l)
        ##################################
        dsp = f'epoch: [{epoch}/{args.nEpochs}]'
        log.info(dsp)
        tqdm_loader.set_description(dsp)

        # TODO: visdom display
        # visdom.line([total_loss], [epoch], win='loss-total', name='total', opts=dict(title='Total-loss'), update='append' if epoch else '')
        # visdom.line([reg_loss], [epoch], win='loss-reg', name='reg', opts=dict(title='Reg-loss'), update='append' if epoch else '')
        # visdom.line([fus_loss], [epoch], win='loss-fus', name='fus', opts=dict(title='Fuse-loss'), update='append' if epoch else '')
        # TODO: save checkpoint
        save_checkpoint(RegNet,  epoch, f'../cache/Joint_reg_fusion/reg_{epoch:04d}.pth') if epoch % interval == 0 else None
        # save_checkpoint(FuseNet, epoch, cache / f'fus_{epoch:04d}.pth') if epoch % interval == 0 else None

def Joint_train(args, tqdm_loader, optimizer_reg, optimizer_fus, RegNet, FuseNet, criterion_reg, criterion_fus, epoch, elastic, affine):

    RegNet.train()
    FuseNet.train()
    DIDF_Encoder.train()
    DIDF_Decoder.train()
    BaseFuseLayer.train()
    DetailFuseLayer.train()

    # TODO: update learning rate of the optimizer
    lr_R = adjust_learning_rate(args, optimizer_reg, epoch - 1)
    lr_F = adjust_learning_rate(args, optimizer_fus, epoch - 1)
    print("Epoch={}, lr_R={}, lr_F={} ".format(epoch, lr_R, lr_F))

    loss_total, loss_reg, loss_fus = [], [], []
    for (ir, it, vi, ir_map, vi_map), _ in tqdm_loader:

        ir, it, vi     = ir.cuda(), it.cuda(), vi.cuda()
        ir_map, vi_map = ir_map.cuda(), vi_map.cuda()

        # TODO: generate warped ir images shengchengbianxingtuxiang
        ir_affine, ir_affine_disp   = affine(ir)
        ir_elastic, ir_elastic_disp = elastic(ir_affine)
        disp_ir = ir_affine_disp + ir_elastic_disp  # cumulative disp grid [batch_size, height, weight, 2]
        ir_warp = ir # ir_elastic

        ir_warp.detach_()
        disp_ir.detach_()

        # TODO: train registration
        ir_pred, ir_f_warp, ir_flow, ir_int_flow1, ir_int_flow2, ir_disp_pre = RegNet(it, ir_warp, vi)
        reg_loss, vgg, ncc, grad = criterion_reg(ir_pred, ir_f_warp, it, ir_warp, ir_flow, ir_int_flow1, ir_int_flow2)

        # TODO: train fusion
        fuse_out  = FuseNet(ir_pred, vi)
        fuse_loss = criterion_fus(fuse_out, ir_pred, vi, ir_map, vi_map)
        # TODO: total loss
        loss = 1.0 * reg_loss + 1.0 * fuse_loss

        optimizer_reg.zero_grad()
        optimizer_fus.zero_grad()
        loss.backward()
        optimizer_reg.step()
        optimizer_fus.step()

        if tqdm_loader.n % 40 == 0:
            show = torch.stack([it[0], ir_warp[0], ir_pred[0], vi[0], fuse_out[0]])
            # visdom.images(show, win='Reg+Fusion')

        loss_total.append(loss.item())
        loss_reg.append(reg_loss.item())
        loss_fus.append(fuse_loss.item())

    l = len(loss_total)
    return sum(loss_total) / l, sum(loss_reg) / l, sum(loss_fus) / l

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def _warp_Dense_loss_unsupervised(criterion, im_pre, im_fwarp, im_fix, im_warp, flow, flow1, flow2):
    total_loss, multi, ncc, grad = criterion(im_pre, im_fwarp, im_fix, im_warp, flow, flow1, flow2)

    return multi, ncc, grad, total_loss

def save_checkpoint(net, epoch, cache):
    model_folder = cache
    model_out_path = model_folder + f'/cp_{epoch:04d}.pth'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = hyper_args()
    visdom = visdom.Visdom(port=8097, env='Reg+Fusion')

    main(args, visdom)