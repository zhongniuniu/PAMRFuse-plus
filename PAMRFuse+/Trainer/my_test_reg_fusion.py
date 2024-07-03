import sys

sys.path.append("..")

import argparse
import pathlib
import warnings
import statistics
import time

import cv2
import numpy as np
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
from torch import Tensor
from tqdm import tqdm
import os

from dataloader.joint_data import JointTestData

from models.deformable_net import DeformableNet
from models.fusion_net import FusionNet

from Trainer.net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0,4"

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RegNet & FuseNet eval process')
    # dataset
    parser.add_argument('--it',   default='../P2M/results/PATMRI/test_latest/images', type=pathlib.Path)
    parser.add_argument('--ir',   default='../datasets/PATMRI/trainB',    type=pathlib.Path)
    parser.add_argument('--vi',   default='../datasets/PATMRI/trainA',      type=pathlib.Path)
    parser.add_argument('--disp', default='../dataset/raw/ctest/Road/disp',    type=pathlib.Path)
    # checkpoint and save path
    parser.add_argument('--ckpt_reg', default='../cache/reg.pth', help='weight checkpoint', type=pathlib.Path) # weight/default.pth
    parser.add_argument('--dst_reg',  default='../result/Deformable/', help='fuse image save folder', type=pathlib.Path)

    parser.add_argument('--ckpt_fus', default='*', help='weight checkpoint', type=pathlib.Path)  # weight/default.pth
    parser.add_argument('--dst_fus',  default='../result/fusion/',help='fuse image save folder', type=pathlib.Path)
    # setup
    parser.add_argument('--dim', default=64, type=int, help='AFuse feather dim')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")

    args = parser.parse_args()
    return args

def main(args):
    ckpt_path = r"../cache/Fuse.pth"
    model_name = "CDDFuse"

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    print("===> Loading datasets")
    data = JointTestData(args.ir, args.it, args.vi, args.disp)
    test_data_loader = torch.utils.data.DataLoader(data, 1, True, pin_memory=True)

    print("===> Building model")
    RegNet = DeformableNet().to(device)
    # FuseNet = FusionNet(nfeats=args.dim).to(device)

    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    print("===> loading trained RegNet model '{}'".format(args.ckpt_reg))
    model_state_dict_reg = torch.load(args.ckpt_reg)
    RegNet.load_state_dict(model_state_dict_reg)

    print("===> loading trained FuseNet model '{}'".format(ckpt_path))
    # model_state_dict_fus = torch.load(args.ckpt_fus)
    # FuseNet.load_state_dict(model_state_dict_fus)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])

    print("===> Starting Testing")
    test(RegNet, Encoder, Decoder, BaseFuseLayer, DetailFuseLayer, test_data_loader, args.dst_reg, args.dst_fus, device)

    pass

def test(RegNet, Encoder, Decoder, BaseFuseLayer, DetailFuseLayer, test_data_loader, dst_reg, dst_fus, device):
    RegNet.eval()
    # FuseNet.eval()
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    full_time = []
    tqdm_loader = tqdm(test_data_loader, disable=True)
    for (ir, it, vi), (ir_path, it_path, vi_path) in tqdm_loader:
        name, ext = os.path.splitext(os.path.basename(ir_path[0]))
        file_name = name + ext
        ir = ir.cuda()
        it = it.cuda()
        vi = vi.cuda()
        # disp = disp.squeeze(0).cuda()  # torch.Size([1, 2, 256, 256])

        # Registration & Fusion
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        start = time.time()
        with torch.no_grad():
            ir_pred, f_warp, flow, int_flow1, int_flow2, disp_pred = RegNet(it, ir, vi)
            torch.set_printoptions(profile="full")
            # print ("ir_pred***************************")
            # print (ir_pred * 255)
            # print ("vi***************************")
            # print (vi)
            # fuse_out = FuseNet(ir_pred, vi)
            feature_V_B, feature_V_D, feature_V = Encoder(f_warp)
            feature_I_B, feature_I_D, feature_I = Encoder(ir)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            data_Fuse, _ = Decoder(f_warp, feature_F_B, feature_F_D)
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            # fi = np.squeeze((data_Fuse * 255).cpu().numpy())
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        end = time.time()
        full_time.append(end - start)

        _, _, h, w = ir.shape
        grid = kornia.utils.create_meshgrid(h, w, device=ir.device).to(ir.dtype)
        grid = grid.permute(0, 3, 1, 2)

        # TODO: Draw grid image
        img_grid = _draw_grid(ir.squeeze().cpu().numpy(), 24)

        # TODO: get warped grid & warped image
        new_grid = grid
        new_grid = new_grid.permute(0, 2, 3, 1)

        warp_grid = torch.nn.functional.grid_sample(img_grid.unsqueeze(0), new_grid, padding_mode='border', align_corners=False)
        warp_combine = 0.8 * ir + 0.2 * warp_grid
        warp_combine = torch.clamp(warp_combine, 0, 1)

        # TODO: get registrated grid & registrated image
        pred_grid = torch.nn.functional.grid_sample(warp_grid, disp_pred, padding_mode='border', align_corners=True)
        pred_combine = 0.8 * ir_pred + 0.2 * pred_grid
        pred_combine = torch.clamp(pred_combine, 0, 1)

        # TODO: save registrated images
        imsave(ir, dst_reg / 'ir', file_name)
        imsave(it, dst_reg / 'it' / file_name)
        imsave(ir_pred, dst_reg / 'ir_reg', file_name)
        imsave(f_warp, dst_reg / 'vi_reg', file_name)
        imsave(img_grid, dst_reg / 'grid', file_name)

        imsave(warp_grid, dst_reg / 'warp_grid', file_name)
        imsave(pred_grid, dst_reg / 'reg_grid', file_name)

        imsave(warp_combine, dst_reg / 'ir_warp_grid', file_name)
        imsave(pred_combine, dst_reg / 'ir_reg_grid', file_name)
        save_flow(flow, dst_reg / 'ir_flow', file_name)
        # save_flow(-disp, dst_reg / 'disp', file_name)  # .permute(0, 3, 1, 2)

        # TODO: save fused images
        imsave(data_Fuse, dst_fus / 'fused' / file_name)
        imsave(ir, dst_fus / 'ir' / file_name)
        imsave(vi, dst_fus / 'vi' / file_name)

    # statistics time record
    full_mean = statistics.mean(full_time[1:])
    print('fuse time (average): {:.4f}'.format(full_mean))
    print('fps (equivalence): {:.4f}'.format(1. / full_mean))

    pass

def _draw_grid(im_cv, grid_size: int = 24):
    im_gd_cv = np.full_like(im_cv, 255.0)
    im_gd_cv = cv2.cvtColor(im_gd_cv, cv2.COLOR_GRAY2BGR)

    height, width = im_cv.shape
    color = (0, 0, 255)
    for x in range(0, width - 1, grid_size):
        cv2.line(im_gd_cv, (x, 0), (x, height), color, 1, 1) # (0, 0, 0)
    for y in range(0, height - 1, grid_size):
        cv2.line(im_gd_cv, (0, y), (width, y), color, 1, 1)
    im_gd_ts = kornia.utils.image_to_tensor(im_gd_cv / 255.).type(torch.FloatTensor).cuda()
    return im_gd_ts

def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
    """
    save images to path
    :param im_s: image(s)
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """

    im_s = im_s if type(im_s) == list else [im_s]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze()
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts)
        im_cv = im_cv * 255.
        cv2.imwrite(str(p), im_cv)

def save_flow(flow: [Tensor], dst: pathlib.Path, im_name: str = ''):
    rgb_flow = flow2rgb(flow, max_value=None) # (3, 512, 512) type; numpy.ndarray
    im_s = rgb_flow if type(rgb_flow) == list else [rgb_flow]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = (im_ts * 255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imwrite(str(p), im_cv)

def flow2rgb(flow_map: [Tensor], max_value: None):
    flow_map_np = flow_map.squeeze().detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    rgb_flow = rgb_map.clip(0, 1)
    return rgb_flow


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = hyper_args()
    main(args)