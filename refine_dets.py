from __future__ import absolute_import, division, print_function

import sys
import os
import os.path as osp
import pprint
import time

import numpy as np
import torch
import yaml
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sacred import Experiment
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.utils import interpolate, plot_sequence
from refiner import Refiner

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor-DPM.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_network_config'])
ex.add_config(ex.configurations[0]._conf['tracktor']['obj_detect_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')

# Tracker = ex.capture(Tracker, prefix='tracker.tracker')

@ex.automain
def my_main(tracktor, prev_result_dir, _config):

    # get previous output
    if not osp.isdir(prev_result_dir):
        raise Exception('Target DIR [%s] dose not exist.'%prev_result_dir)

    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    print("[*] Building object detector")
    if tracktor['network'].startswith('frcnn'):
        # FRCNN
        from tracktor.frcnn import FRCNN
        from frcnn.model import config

        if _config['frcnn']['cfg_file']:
            config.cfg_from_file(_config['frcnn']['cfg_file'])
        if _config['frcnn']['set_cfgs']:
            config.cfg_from_list(_config['frcnn']['set_cfgs'])

        obj_detect = FRCNN(num_layers=101)
        obj_detect.create_architecture(2, tag='default',
            anchor_scales=config.cfg.ANCHOR_SCALES,
            anchor_ratios=config.cfg.ANCHOR_RATIOS)
        obj_detect.load_state_dict(torch.load(tracktor['obj_detect_weights']))
    elif tracktor['network'].startswith('fpn'):
        # FPN
        from tracktor.fpn import FPN
        from fpn.model.utils import config
        config.cfg.TRAIN.USE_FLIPPED = False
        config.cfg.CUDA = True
        config.cfg.TRAIN.USE_FLIPPED = False
        checkpoint = torch.load(tracktor['obj_detect_weights'])

        if 'pooling_mode' in checkpoint.keys():
            config.cfg.POOLING_MODE = checkpoint['pooling_mode']

        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                    'ANCHOR_RATIOS', '[0.5,1,2]']
        config.cfg_from_file(_config['tracktor']['obj_detect_config'])
        config.cfg_from_list(set_cfgs)

        obj_detect = FPN(('__background__', 'pedestrian'), 101, pretrained=False)
        obj_detect.create_architecture()

        obj_detect.load_state_dict(checkpoint['model'])
    else:
        raise NotImplementedError(f"Object detector type not known: {tracktor['network']}")

    pprint.pprint(config.cfg)
    obj_detect.eval()
    obj_detect.cuda()

    refiner = Refiner(obj_detect)

    print("[*] Beginning evaluation...")

    time_total = 0
    for sequence in Datasets(tracktor['dataset']):
        refiner.reset(sequence, tracktor['dataset'], prev_result_dir)

        now = time.time()

        print("[*] Evaluating: {}".format(sequence))

        data_loader = DataLoader(sequence, batch_size=1, shuffle=False)
        for i, frame in enumerate(data_loader):
            if i >= len(sequence) * tracktor['frame_split'][0] and i <= len(sequence) * tracktor['frame_split'][1]:
                refiner.step(frame)
        results = refiner.get_results()

        time_total += time.time() - now

        print("[*] Tracks found: {}".format(len(results)))
        print("[*] Time needed for {} evaluation: {:.3f} s".format(sequence, time.time() - now))

        if tracktor['interpolate']:
            results = interpolate(results)

        sequence.write_results(results, osp.join(output_dir))

        if tracktor['write_images']:
            plot_sequence(results, sequence, osp.join(output_dir, tracktor['dataset'], str(sequence)))

    print("[*] Evaluation for all sets (without image generation): {:.3f} s".format(time_total))
