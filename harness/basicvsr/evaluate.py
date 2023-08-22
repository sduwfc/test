import argparse
import os
from threading import Condition

try:
    from inference import init_logger, BasicVSR, get_image_list_with_name
    from metrics import psnr, ssim
    from misc import tensor2img
except ImportError:
    from .inference import init_logger, BasicVSR, get_image_list_with_name
    from .metrics import psnr, ssim
    from .misc import tensor2img


import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="BasicVSR Inference")
    parser.add_argument("lrdir", help="Low resolution dir")
    parser.add_argument("gtdir", help="Ground truth dir")
    parser.add_argument(
        "--spynet-model",
        default="models/spynet.compilation/compilation.bmodel",
        help="SPyNet model path",
    )
    parser.add_argument(
        "--backward-residual-model",
        default="models/backward_residual.compilation/compilation.bmodel",
        help="Backward residual model path",
    )
    parser.add_argument(
        "--forward-residual-model",
        default="models/forward_residual.compilation/compilation.bmodel",
        help="Forward residual model path",
    )
    parser.add_argument(
        "--upsample-model",
        default="models/upsample.compilation/compilation.bmodel",
        help="Upsample model path",
    )
    parser.add_argument("--width", default=176, help="resize width")
    args = parser.parse_args()
    return args


def run_one(model_path, lrdir, gtdir, devices, logger, width=176):
    spynet_model = model_path["spynet"]
    backward_residual_model = model_path["backward"]
    forward_residual_model = model_path["forward"]
    upsample_model = model_path["upsample"]

    clips = os.listdir(lrdir)
    results = [[] for i in clips]
    index = 0
    cv = Condition()

    def result_callback(out, info):
        img = tensor2img(torch.from_numpy(out))
        fn = os.path.basename(info["filename"])
        gtfn = os.path.join(gtdir, info["clip"], fn)
        gt = cv2.imread(gtfn)
        gt = cv2.resize(gt, (width * 4, 576))
        results[info["clip_index"]].append(
            [psnr(img, gt, convert_to="y"), ssim(img, gt, convert_to="y")]
        )
        nonlocal index
        index += 1
        with cv:
            cv.notify()

    basicvsr = BasicVSR(
        spynet_model,
        backward_residual_model,
        forward_residual_model,
        upsample_model,
        logger=logger,
        result_callback=result_callback,
        devices=devices,
    )

    import time

    total_lr_num = 0
    for i, clip in enumerate(clips):
        lrs, filenames = get_image_list_with_name(
            os.path.join(lrdir, clip), width=width
        )
        total_lr_num += len(lrs)
        basicvsr.put(lrs, clip=clip, clip_index=i, filenames=filenames)
        with cv:
            while index < total_lr_num:
                cv.wait()
    basicvsr.join()
    psnr_clip_avgs = []
    ssim_clip_avgs = []
    res = {}
    for i, clip in enumerate(clips):
        avg_psnr = np.mean([r[0] for r in results[i]])
        avg_ssim = np.mean([r[1] for r in results[i]])
        psnr_clip_avgs.append(avg_psnr)
        ssim_clip_avgs.append(avg_ssim)
        res[f"psnr_{clip}"] = avg_psnr
        res[f"ssim_{clip}"] = avg_ssim
    psnr_avg = np.mean(psnr_clip_avgs)
    ssim_avg = np.mean(ssim_clip_avgs)

    res["psnr_all"] = psnr_avg
    res["ssim_all"] = ssim_avg

    return res


def run(spynet, backward, forward, upsample, datadir, key, devices, kwargs):
    gtdir = os.path.join(datadir, "GT")
    lrdir = os.path.join(datadir, key)
    # bddir = os.path.join(datadir,"BDx4")
    # bidir = os.path.join(datadir,"BIx4")

    model_path = dict(
        spynet=spynet,
        backward=backward,
        forward=forward,
        upsample=upsample,
    )
    logger = init_logger()
    res = run_one(model_path, lrdir, gtdir, devices, logger=logger)
    return res


def main():
    logger = init_logger()
    args = parse_args()

    clips = os.listdir(args.lrdir)
    results = [[] for i in clips]
    index = 0
    cv = Condition()

    def result_callback(out, info):
        img = tensor2img(torch.from_numpy(out))
        fn = os.path.basename(info["filename"])
        gtfn = os.path.join(args.gtdir, info["clip"], fn)
        gt = cv2.imread(gtfn)
        gt = cv2.resize(gt, (704, 576))
        results[info["clip_index"]].append(
            [psnr(img, gt, convert_to="y"), ssim(img, gt, convert_to="y")]
        )
        nonlocal index
        index += 1
        with cv:
            cv.notify()

    basicvsr = BasicVSR(
        args.spynet_model,
        args.backward_residual_model,
        args.forward_residual_model,
        args.upsample_model,
        result_callback,
        logger,
    )
    import time

    start = time.time()
    total_lr_num = 0
    for i, clip in enumerate(clips):
        lrs, filenames = get_image_list_with_name(os.path.join(args.lrdir, clip))
        total_lr_num += len(lrs)
        basicvsr.put(lrs, clip=clip, clip_index=i, filenames=filenames)
        with cv:
            while index < total_lr_num:
                cv.wait()
    basicvsr.join()
    latency = time.time() - start
    logger.info("{:.2f}ms per frame".format(latency * 1000 / total_lr_num))
    psnr_clip_avgs = []
    ssim_clip_avgs = []
    for i, clip in enumerate(clips):
        avg_psnr = np.mean([r[0] for r in results[i]])
        avg_ssim = np.mean([r[1] for r in results[i]])
        psnr_clip_avgs.append(avg_psnr)
        ssim_clip_avgs.append(avg_ssim)
        logger.info("{} psnr {:.4f}, ssim {:.4f}".format(clip, avg_psnr, avg_ssim))
    psnr_avg = np.mean(psnr_clip_avgs)
    ssim_avg = np.mean(ssim_clip_avgs)
    logger.info("total psnr {:.4f}, ssim {:.4f}".format(psnr_avg, ssim_avg))


def main2():
    logger = init_logger()
    args = parse_args()

    res = run_one(
        dict(
            spynet=args.spynet_model,
            backward=args.backward_residual_model,
            forward=args.forward_residual_model,
            upsample=args.upsample_model,
        ),
        args.lrdir,
        args.gtdir,
        devices=None,
        width=args.width,
        logger=logger,
    )


if __name__ == "__main__":
    main2()
