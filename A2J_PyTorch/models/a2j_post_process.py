import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class A2JPostProcess(nn.Module):
    def __init__(self, P_h=[2,6], P_w=[2,6], shape=[48,26], stride=8, is_3D=True):
        super(A2JPostProcess, self).__init__()

        anchor_cluster = util.generate_anchor_cluster(P_h=P_h, P_w=P_w)
        anchor_coords = util.replicate(anchor_cluster, shape, stride)
        anchor_coords = torch.from_numpy(anchor_coords).float().cuda()

        self.anchor_coords = anchor_coords
        self.is_3D = is_3D

    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0], b.shape[0]).cuda()
        for i in range(a.shape[1]):
            dis += torch.pow(torch.unsqueeze(a[:,i], dim=1) - b[:,i], 0.5)
        return dis

    def forward(self, heads):
        if self.is_3D:
            responses, offsets, depths = heads
        else:
            responses, offsets = heads

        results = []
        batch_size = responses.shape[0]
        for j in range(batch_size):
            # responses has shape N*(w*h*A)*P
            response = responses[j]
            # softmax response to get heatmap, then create 1 and 2 channel versions of it
            heatmap_1c = F.softmax(response, dim=0)#(w*h*A)*P
            heatmap_2c = heatmap_1c.expand(heatmap_1c.shape[0], heatmap_1c.shape[1], 2)#(w*h*A)*P*2
            # offsets has shape N*(w*h*A)*P*2
            joint_coords = torch.unsqueeze(self.anchor_coords, 1) + offsets[j]
            joint_coords = (heatmap_2c * joint_coords).sum(0)

            if self.is_3D:
                # depths has shape N*(w*h*A)*P
                depth = (heatmap_1c * depths[j]).sum(0)
                # depth = torch.unsqueeze(depth, 1)
                # update joint_coords to include depth
                joint_coords = torch.cat((joint_coords, depth), 1)

            results.append(joint_coords)

        return torch.stack(results)
