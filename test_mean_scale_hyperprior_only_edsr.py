
# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from collections import defaultdict
import argparse
import math
import random
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.zoo import image_models
from dataset import Kodak24Dataset
from load_model import load_model
from edsr import EDSR
import torch.nn.functional as F

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    '''
    "bmshj2018-factorized": bmshj2018_factorized,
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020-anchor": cheng2020_anchor,
    "cheng2020-attn": cheng2020_attn,
    '''
    parser.add_argument(
        "-m",
        "--model",
        default="mbt2018-mean",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=0, help="quality of the model"
    )

    parser.add_argument(
        "-d", "--dataset", type=str, default='../data', help="Training dataset"
    )
    parser.add_argument(
        "-save_dir", "--save_dir", type=str, default='save/', help="save_dir"
    )
    parser.add_argument(
        "-log_dir", "--log_dir", type=str, default='log/', help="log_dir"
    )
    parser.add_argument(
        "-total_step",
        "--total_step",
        default=5000000,    # infinite (we can use early stop)
        type=int,
        help="total_step (default: %(default)s)",
    )
    parser.add_argument(
        "-test_step",
        "--test_step",
        default=5000,
        type=int,
        help="test_step (default: %(default)s)",
    )
    parser.add_argument(
        "-save_step",
        "--save_step",
        default=100000,
        type=int,
        help="save_step (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    # parser.add_argument(
    #     "--lambda",
    #     dest="lmbda",
    #     type=float,
    #     default=1e-2,
    #     # quality   :     1        2        3        4        5        6       7       8
    #     # lambda    :  0.0018   0.0035   0.0067   0.0130   0.0250   0.0483  0.0932   0.1800
    #     help="Bit-rate distortion parameter (default: %(default)s)",
    # )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=123, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def build_dataset(args):
    # Warning, the order of the transform composition should be kept.
    kodak_transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    kodak_dataset = Kodak24Dataset(
        args.dataset,
        transform=kodak_transform,
    )

    kodak_dataloader = DataLoader(
        kodak_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return kodak_dataloader


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#test 하는 코드
def comrpess_and_decompress(model, edsr, test_dataloader, device):
    psnr = AverageMeter()
    bpp = AverageMeter()
    with torch.no_grad():
        for i_, x in enumerate(test_dataloader):
            x = x.to(device)

            # Downsampling
            x_lr = F.interpolate(x, scale_factor=0.5, mode='bicubic').clamp_(0, 1)

            # compress
            compressed = model.compress(x_lr)  # {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
            strings = compressed['strings']
            shape = compressed['shape']

            # decompress
            decompressed = model.decompress(strings, shape)

            # Upsampling
            x_hat = edsr(decompressed['x_hat']).clamp_(0, 1)

            bpp_y = (len(strings[0][0])) * 8 / (x.shape[2] * x.shape[3]) #string 이 8비트라 8곱함
            bpp_z = (len(strings[1][0])) * 8 / (x.shape[2] * x.shape[3])
            bpp_ = bpp_y + bpp_z
            mse_ = (x_hat - x).pow(2).mean()
    
            psnr_ = 10 * (torch.log(1 * 1 / mse_) / math.log(10))
            bpp.update(bpp_)
            psnr.update(psnr_)
            print(
                f"\tPSNR: {psnr_:.3f} |"
                f"\tBPP: {bpp_:.3f} |"
            )

    print(
        f"\tTest PSNR: {psnr.avg:.3f} |"
        f"\tTest BPP: {bpp.avg:.3f} |"
    )

def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # device = "cpu"
    device = "cuda"
    model = load_model(args.model, metric="mse", quality=args.quality, pretrained=True).to(device).eval()
    edsr = EDSR().to(device).eval()
    # edsr.load_state_dict(torch.load('./save/EDSR/edsr_baseline_x2-1bc95232.pt'), strict=False)

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        edsr.load_state_dict(checkpoint["state_dict_edsr"])

    test_dataloader = build_dataset(args)
    comrpess_and_decompress(model, edsr, test_dataloader, device)


if __name__ == "__main__":
    main(sys.argv[1:])
