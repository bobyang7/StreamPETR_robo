import os
from tqdm import tqdm
import json
import multiprocessing
from visual_nuscenes import NuScenes
use_gt = False
res_dir = "test/cfg_self_history/Thu_Aug__3_02_50_25_2023/pts_bbox"
out_dir = os.path.join(res_dir, 'visualize')
result_json = os.path.join(res_dir, 'results_nusc')
dataroot='./data/nuscenes'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)

with open(f"{result_json}.json") as f:
    table = json.load(f)
tokens = list(table['results'].keys())

def vis_token(token):
    if use_gt:
        nusc.render_sample(token, out_path = f"{out_dir}/"+token+"_gt.png", verbose=False)
    else:
        nusc.render_sample(token, out_path = f"{out_dir}/"+token+"_pred.png", verbose=False)

vis_cnt = 100
num_worker = 8
tokens = tokens[:vis_cnt]
with multiprocessing.Pool(num_worker) as pool:
    for item in tqdm(
        pool.imap(vis_token, tokens),
        total=len(tokens)
    ):
        pass

