import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    
    parser.add_argument('--h', type=int, default=1024, help='height')
    parser.add_argument('--w', type=int, default=2048, help='weight')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
     # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # model = MMDataParallel(model, device_ids=[0])
    model.cuda()
    model.eval()
    # the first several iterations may be very slow so skip them
    num_warmup = 20
    pure_inf_time = 0
    total_iters = 100
    dummy = torch.randn(1, 3, args.w, args.h).cuda()
    for j in range(total_iters + num_warmup):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            model.forward_dummy(dummy)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if (j + 1) > num_warmup:
            pure_inf_time += elapsed
        
    fps = total_iters / pure_inf_time
    print(f'fps: {fps:.2f} img / s')

        


if __name__ == '__main__':
    main()
