import torch
import math
torch.pi = math.pi

def normalize_bbox_old(bboxes, pc_range):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def normalize_bbox_bin(bboxes, pc_range, rot_seg_num=4, rot_seg_offset=0, rot_seg_overlap=torch.pi / 4):
    """Normalize bbox."""
    c_x = bboxes[..., 0:1]
    c_y = bboxes[..., 1:2]
    c_z = bboxes[..., 2:3]
    size_w = bboxes[..., 3:4].log()
    size_l = bboxes[..., 4:5].log()
    size_h = bboxes[..., 5:6].log()
    rot_segments = (
        torch.pi
        * 2
        / rot_seg_num
        * torch.arange(torch.pi * 2 / rot_seg_num + rot_seg_overlap * 2).cuda()
        + rot_seg_offset
        - rot_seg_overlap
    )
    rot = bboxes[..., 6:7]
    rot_offset = (rot - rot_segments[None]) % (2 * torch.pi)
    rot_cls = (rot_offset > 0) & (
        rot_offset < torch.pi * 2 / rot_seg_num + rot_seg_overlap * 2
    )
    if bboxes.size(-1) > 7:
        v_x = bboxes[..., 7:8]
        v_y = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (
                c_x,
                c_y,
                c_z,
                size_w,
                size_l,
                size_h,
                torch.stack((rot_offset.sin(), rot_offset.cos()), dim=-1).reshape(
                    *bboxes.shape[:-1], -1
                ),
                rot_cls,
                v_x,
                v_y,
            ),
            dim=-1,
        )
    else:
        normalized_bboxes = torch.cat(
            (
                c_x,
                c_y,
                size_l,
                size_w,
                c_z,
                size_h,
                torch.stack((rot_offset.sin(), rot_offset.cos()), dim=-1).reshape(
                    *bboxes.shape[:-1], -1
                ),
                rot_cls,
            ),
            dim=-1,
        )
    return normalized_bboxes

def normalize_bbox(bboxes, pc_range):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    # 输入[n,1], 输出[n,3]

    phase_shift_targets = tuple(
    torch.cos(rot + 2 * torch.pi * x / 3)
    for x in range(3))

    enc_rot = torch.cat(phase_shift_targets, axis=-1)

    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, enc_rot, vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, enc_rot), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox_old(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]

    # size
    w = normalized_bboxes[..., 3:4]
    l = normalized_bboxes[..., 4:5]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

# xyz wlh 8 4 vxvy
def denormalize_bbox_bin(
    normalized_bboxes,
    pc_range,
    rot_seg_num=4,
    rot_seg_offset=0,
    rot_seg_overlap=torch.pi / 4,
):
    """Denormalize bbox."""
    split_sections = (1,) * 6 + (rot_seg_num * 2, rot_seg_num) + (1,) * 2
    
    (c_x, c_y, c_z, size_w, size_l, size_h, rot_sine_cosin, rot_cls, vx, vy) = torch.split(
        normalized_bboxes,
        split_size_or_sections=split_sections,
        dim=-1,
    )
    rot_cls = torch.argmax(rot_cls, dim=-1)
    rot_base = torch.pi * 2 / rot_seg_num * rot_cls + rot_seg_offset - rot_seg_overlap
    rot_sine, rot_cosine = torch.split(
        torch.index_select(
            rot_sine_cosin.reshape(-1, 2),
            dim=0,
            index=torch.arange(normalized_bboxes.shape[0]).cuda() * rot_seg_num
            + rot_cls,
        ),
        split_size_or_sections=(1, 1),
        dim=-1,
    )
    rot = (
        torch.clamp(
            torch.atan2(rot_sine, rot_cosine),
            max=torch.pi * 2 / rot_seg_num + rot_seg_overlap * 2,
            min=0,
        )
        + rot_base.unsqueeze(-1)
    ) % (torch.pi * 2)

    size_w = size_w.exp()
    size_l = size_l.exp()
    size_h = size_h.exp()

    denormalized_bboxes = torch.cat(
        [c_x, c_y, c_z, size_w, size_l, size_h, rot, vx, vy], dim=-1
    )
    return denormalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_enc = normalized_bboxes[..., 6:9]

    coef_sin = torch.tensor(
        tuple(
            torch.sin(torch.tensor(2 * k * torch.pi / 3))
            for k in range(3))).to(rot_enc)
    coef_cos = torch.tensor(
        tuple(
            torch.cos(torch.tensor(2 * k * torch.pi / 3))
            for k in range(3))).to(rot_enc)

    phase_sin = torch.sum(
        angle_preds[:, 0: 3] * coef_sin,
        dim=-1,
        keepdim=keepdim)
    phase_cos = torch.sum(
        angle_preds[:, 0: 3] * coef_cos,
        dim=-1,
        keepdim=keepdim)
    phase_mod = phase_cos**2 + phase_sin**2
    phase = -torch.atan2(phase_sin, phase_cos)  # In range [-pi,pi)

    # Set the angle of isotropic objects to zero
    phase[phase_mod < self.thr_mod] *= 0
    rot = phase

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]

    # size
    w = normalized_bboxes[..., 3:4]
    l = normalized_bboxes[..., 4:5]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes
