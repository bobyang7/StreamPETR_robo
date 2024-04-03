import json
import os

# please specify your path here
folder = "/home/bo.yang5/streampetr/test/repdetr3d_eva02_800_bs2_seq_24e_robo"
files = os.listdir(folder)

all_dict = {}
for file in files:
    path = os.path.join(folder, file, "pts_bbox", "results_nusc.json")
    with open(path, "r") as f:
        data = json.load(f)
        data['meta']['use_external'] = False
    all_dict[file] = data

with open("/home/bo.yang5/streampetr/pred.json", "w") as f:
    json.dump(all_dict, f)
    a = 1
