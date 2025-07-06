import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

from utils.common import get_model
from utils.config import ConfigDict
from data.dataset import LaneTestDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Lane Detection Demo')
    parser.add_argument('--dataset', type=str, default='culane', choices=['culane', 'tusimple'], help='Dataset to use for inference')
    parser.add_argument('--model', type=str, default='res18', choices=['res18', 'res34'], help='Model architecture to use')
    parser.add_argument('--image_dir', type=str, default='images/', help='Directory containing images for inference')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='Directory to save output images')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/', help='Path to the model checkpoint directory')
    return parser.parse_args()

def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = ConfigDict({
        'dataset': args.dataset
    })
    cfg.batch_size = 1

    if (args.dataset == 'culane'):
        cls_num_per_lane = 18
        cfg.num_row = 72
        cfg.num_col = 81
        cfg.num_cell_row = 200
        cfg.num_cell_col = 100
        cfg.num_lanes = 4
        cfg.use_aux = False
        cfg.train_width = 1600
        cfg.train_height = 320
        cfg.fc_norm = True
        cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
        cfg.crop_ratio = 0.6
    elif (args.dataset == 'tusimple'):
        cls_num_per_lane = 56
        cfg.num_row = 56
        cfg.num_col = 41
        cfg.num_cell_row = 100
        cfg.num_cell_col = 100
        cfg.num_lanes = 4
        cfg.use_aux = False
        cfg.train_width = 800
        cfg.train_height = 320
        cfg.fc_norm = False
        cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
        cfg.crop_ratio = 0.8
    else:
        raise NotImplementedError
    
    if (args.model == 'res18'):
        cfg.backbone = '18'
    elif (args.model == 'res34'):
        cfg.backbone = '34'
    else:
        raise NotImplementedError
    
    net = get_model(cfg)

    ckpt_path = os.path.join(args.ckpt_dir, f'{args.dataset}_{args.model}.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist. Please provide a valid model checkpoint.")
    state_dict = torch.load(ckpt_path, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=True)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = LaneTestDataset(
        path=args.image_dir, img_transform=img_transforms, crop_size=cfg.train_height
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=1)

    for i, data in enumerate(tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            pred = net(imgs)

        vis = cv2.imread(os.path.join(args.image_dir, names[0]))
        img_h, img_w = vis.shape[:2]
        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=img_w, original_image_height=img_h)
        for lane in coords:
            for x, y in lane:
                cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
        output_path = os.path.join(args.output_dir, names[0])
        cv2.imwrite(output_path, vis)