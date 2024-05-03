# Train & inference
## Train
Train the repdetr3d model following:

```bash
tools/dist_train.sh projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_nuscenes.py 8 --work-dir work_dirs/repdetr3d_eva02_800_bs2_seq_24e_nuscenes
```
Train the repdetr3d+depth model following:

```bash
tools/dist_train.sh projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_nuscenes_depth_psc.py 8 --work-dir work_dirs/repdetr3d_eva02_800_bs2_seq_24e_nuscenes_depth_psc
```

## Test on Nuscenes
Test the repdetr3d model following:

```bash
tools/dist_test.sh projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_nuscenes.py  work_dirs/repdetr3d_eva02_800_bs2_seq_24e/latest.pth  8 --eval bbox
```

Test the repdetr3d+depth model following:
```bash
tools/dist_test.sh projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_nuscenes_depth_psc.py work_dirs/repdetr3d_eva02_800_bs2_seq_24e_robo_custom_single/latest.pth  8 --eval bbox
```

## Test on Corruptions
Test the repdetr3d model following:

```bash
tools/dist_test.sh projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_robo.py  work_dirs/repdetr3d_eva02_800_bs2_seq_24e/latest.pth  8 --eval bbox
```

Test the repdetr3d+depth model following:
```bash
tools/dist_test.sh projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_robo_depth_psc.py work_dirs/repdetr3d_eva02_800_bs2_seq_24e_robo_custom_single/latest.pth  8 --eval bbox
```
