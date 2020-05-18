import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class A2JLoss(nn.Module):
    def __init__(self, P_h=[2,6], P_w=[2,6], shape=[8,4], stride=8, spatialFactor=0.1, is_3D=True):
        super(A2JLoss, self).__init__()

        anchor_cluster = util.generate_anchor_cluster(P_h=P_h, P_w=P_w)
        anchor_coords = util.replicate(anchor_cluster, shape, stride)
        anchor_coords = torch.from_numpy(anchor_coords).float()#cuda()

        self.anchor_coords = anchor_coords
        self.is_3D = is_3D

        self.spatialFactor = spatialFactor

    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0], b.shape[0]).cuda()
        for i in range(a.shape[1]):
            dis += torch.pow(torch.unsqueeze(a[:,i], dim=1) - b[:,i], 0.5)
        return dis

    def forward(self, heads, annotations):
        alpha = 0.25
        gamma = 2.0

        if self.is_3D:
            responses, offsets, depths = heads
        else:
            responses, offsets = heads

        losses_anchors = []
        losses_joint_coords = []
        batch_size = responses.shape[0]
        for j in range(batch_size):
            '''
            Compute heatmap and joint coords, just like in A2JPostProcess
            '''
            # responses has shape N*(w*h*A)*P
            response = responses[j]
            # softmax response to get heatmap, then create 1 and 2 channel versions of it
            heatmap_1c = F.softmax(response, dim=0)#(w*h*A)*P
            # heatmap_2c = torch.unsqueeze(heatmap_1c, 2).expand(heatmap_1c.shape[0], heatmap_1c.shape[1], 2)#(w*h*A)*P*2
            heatmap_2c = heatmap_1c.expand(heatmap_1c.shape[0], heatmap_1c.shape[1], 2)#(w*h*A)*P*2
            # offsets has shape N*(w*h*A)*P*2
            joint_coords = torch.unsqueeze(self.anchor_coords, 1) + offsets[j,:,:,:]
            joint_coords = (heatmap_2c * joint_coords).sum(0)

            '''
            Now use ground truth to compute losses
            '''
            # annotations has shape N*P*3
            bbox_annotation = annotations[j,:,:]
            gt_xy = bbox_annotation[:,:2]#P*2

            # ##################################################################
            # compute HEATMAP loss (make sure it lights up only near joints)
            diff_anchors = torch.abs(
                gt_xy - (heatmap_2c * torch.unsqueeze(self.anchor_coords, 1)).sum(0)
            )#P*2
            loss_anchors = torch.where(
                torch.le(diff_anchors, 1),
                0.5 * (1.0) * torch.pow(diff_anchors, 2),
                diff_anchors - 0.5 / (1.0)
            )
            losses_anchors.append(loss_anchors.mean())

            # ##################################################################
            # compute JOINT POSITION loss (X and Y axis)
            diff_xy = torch.abs(
                gt_xy - joint_coords
            )#P*2
            loss_xy = torch.where(
                torch.le(diff_xy, 1),
                0.5 * (1.0) * torch.pow(diff_xy, 2),
                diff_xy - 0.5 / (1.0)
            )
            loss_joint_coords = loss_xy.mean() * self.spatialFactor

            # ##################################################################
            # compute JOINT POSITION loss (Z axis)
            if self.is_3D:
                gt_z = bbox_annotation[:,2] #P
                diff_z = torch.abs(
                    gt_z - (heatmap_1c * depths[j,:,:]).sum(0)
                )#P
                loss_z = torch.where(
                    torch.le(diff_z, 3),
                    0.5 * (1/3) * torch.pow(diff_z, 2),
                    diff_z - 0.5 / (1/3)
                )
                loss_joint_coords += loss_z.mean()

            losses_joint_coords.append(loss_joint_coords)
        return torch.stack(losses_anchors).mean(dim=0, keepdim=True), torch.stack(losses_joint_coords).mean(dim=0, keepdim=True)
