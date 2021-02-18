from __future__ import print_function

import os
import cv2
import time
import torch
import random
import shutil
import argparse
import numpy as np
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import is_image, draw_caption, hyp_parse
from utils.utils import show_dota_results
from eval import evaluate
from datasets.DOTA_devkit.ResultMerge_multi_process import ResultMerge

DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'IC13': IC13Dataset,
            'HRSC2016': HRSCDataset,
            'DOTA':DOTADataset,
            'UCAS_AOD':UCAS_AODDataset,
            'NWPU_VHR':NWPUDataset
            }

def generate_colors(dataset):
    num_colors = {'VOC' : 20 ,
            'IC15': 1,
            'IC13': 1,
            'HRSC2016': 1,
            'DOTA':15,
            'UCAS_AOD':2,
            'NWPU_VHR':10
            }
    if num_colors[dataset] == 1:
        colors = [(0, 255, 0)]
    elif num_colors[dataset] == 2:
        colors = [(0, 255, 0), (0, 0, 255)]
    else:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_colors[dataset])]
    return colors


def demo(args):
    hyps = hyp_parse(args.hyp)
    ds = DATASETS[args.dataset](level = 1)
    model = RetinaNet(backbone=args.backbone, hyps=hyps)
    colors = generate_colors(args.dataset)
    if args.weight.endswith('.pth'):
        chkpt = torch.load(args.weight)
        # load model
        if 'model' in chkpt.keys():
            model.load_state_dict(chkpt['model'])
        else:
            model.load_state_dict(chkpt)
        print('load weight from: {}'.format(args.weight))
    model.eval()

    t0 = time.time()
    if not args.dataset == 'DOTA':
        ims_list = [x for x in os.listdir(args.ims_dir) if is_image(x)]
        for idx, im_name in enumerate(ims_list):
            s = ''
            t = time.time()
            im_path = os.path.join(args.ims_dir, im_name)   
            s += 'image %g/%g %s: ' % (idx, len(ims_list), im_path)
            src = cv2.imread(im_path, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            cls_dets = im_detect(model, im, target_sizes=args.target_size)
            for j in range(len(cls_dets)):
                cls, scores = cls_dets[j, 0], cls_dets[j, 1]
                bbox = cls_dets[j, 2:]
                if len(bbox) == 4:
                    draw_caption(src, bbox, '{:1.3f}'.format(scores))
                    cv2.rectangle(src, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)
                else:
                    pts = np.array([rbox_2_quad(bbox[:5]).reshape((4, 2))], dtype=np.int32)
                    cv2.drawContours(src, pts, 0, thickness=2, color=colors[int(cls-1)])
                    put_label = True
                    plot_anchor = False
                    if put_label:
                        label = ds.return_class(cls) + str(' %.2f' % scores)
                        fontScale = 0.45
                        font = cv2.FONT_HERSHEY_COMPLEX
                        thickness = 1
                        t_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=thickness)[0]
                        c1 = tuple(bbox[:2].astype('int'))
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 5
                        # import ipdb;ipdb.set_trace()

                        cv2.rectangle(src, c1, c2, colors[int(cls-1)], -1)  # filled
                        cv2.putText(src, label, (c1[0], c1[1] -4), font, fontScale, [0, 0, 0], thickness=thickness, lineType=cv2.LINE_AA)
                        if plot_anchor:
                            pts = np.array([rbox_2_quad(bbox[5:]).reshape((4, 2))], dtype=np.int32)
                            cv2.drawContours(src, pts, 0, color=(0, 0, 255), thickness=2)
            print('%sDone. (%.3fs) %d objs' % (s, time.time() - t, len(cls_dets)))
            # save image

            out_path = os.path.join('outputs' , os.path.split(im_path)[1])
            cv2.imwrite(out_path, src)
    ## DOTA detct on large image
    else:
        evaluate(args.target_size,
                args.ims_dir,    
                'DOTA',
                args.backbone,
                args.weight,
                hyps = hyps,
                conf = 0.05)
        if  os.path.exists('outputs/dota_out'):
            shutil.rmtree('outputs/dota_out')
        os.mkdir('outputs/dota_out')
        exec('cd outputs &&  rm -rf detections && rm -rf integrated  && rm -rf merged')    
        ResultMerge('outputs/detections', 
                    'outputs/integrated',
                    'outputs/merged',
                    'outputs/dota_out')
        img_path = os.path.join(args.ims_dir,'images')
        label_path = 'outputs/dota_out'
        save_imgs =  False
        if save_imgs:
            show_dota_results(img_path,label_path)
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--weight', type=str, default='weights/last.pth')
    # HRSC
    # parser.add_argument('--dataset', type=str, default='HRSC2016')    
    # parser.add_argument('--ims_dir', type=str, default='HRSC2016/Test') 
    # DOTA 
    # parser.add_argument('--dataset', type=str, default='DOTA')
    # parser.add_argument('--ims_dir', type=str, default='DOTA/test')
    # UCAS-AOD
    parser.add_argument('--dataset', type=str, default='UCAS_AOD')
    parser.add_argument('--ims_dir', type=str, default='UCAS_AOD/Test')  
    # IC13
    # parser.add_argument('--dataset', type=str, default='IC13')
    # parser.add_argument('--ims_dir', type=str, default='ICDAR13/test')
    # NWPU
    # parser.add_argument('--dataset', type=str, default='HRSC2016')
    # parser.add_argument('--ims_dir', type=str, default='HRSC2016/Test')   
    



    parser.add_argument('--target_size', type=int, default=[800])
    demo(parser.parse_args())

