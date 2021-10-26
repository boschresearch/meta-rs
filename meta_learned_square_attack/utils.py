# Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import one_hot

from torch.nn.functional import affine_grid
from torch.nn.functional import grid_sample

import torch.nn.functional as F

class BatchedModel:
    """ Wrapper around a pytorch model that allows batched propagation to control the amount of required GPU memory"""
    def __init__(self, model, batch_size, num_classes):
        self.model = model.to("cuda")
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model.eval()

    def __call__(self, x):
        n_batches = int(np.ceil(x.shape[0] / self.batch_size))
        logits = torch.empty((x.shape[0], self.num_classes)).to("cuda")
        for i_b in range(n_batches):
            x_batch = x[i_b * self.batch_size:(i_b + 1) * self.batch_size]
            output = self.model(x_batch)
            if type(output) is tuple:
                output = output[0]
            logits[i_b * self.batch_size:(i_b + 1) * self.batch_size] = output

        return logits


class File:
    def __init__(self, path):
        self.path = path
        folder = '/'.join(path.split('/')[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)

    def write(self, line):
        with open(self.path, 'a') as f:
            f.write(line + '\n')
            f.flush()


def save_progress_plot(path, n_iter, metrics, labels=None, colors=None, title=None):
    iter_range = np.arange(n_iter)
    plt.figure()
    for i in range(len(metrics)):
        kwargs = {}
        if colors is not None:
            kwargs['color'] = colors[i]
        if labels is not None:
            kwargs['label'] = labels[i]

        plt.plot(iter_range, metrics[i, iter_range], **kwargs)

    plt.title(title)
    if labels is not None:
        plt.legend()
    plt.savefig(path)
    plt.close()


def margin_function(logits, y):
    y_one_hot = one_hot(y, num_classes=logits.shape[1]).bool()
    true_class_logit = (y_one_hot * logits).sum(1, keepdims=True)
    diff = true_class_logit - logits
    diff[y_one_hot] = float('inf')
    return diff.min(dim=1)[0]


def minus_ce(logits, y):
    return -F.cross_entropy(logits, y, reduction="none")


# mimic auto attack paper behavior
def margin_function_auto_attack(logits, y):
    logits_1 = logits.detach().clone()
    u = torch.arange(y.shape[0])
    y_corr = logits_1[u, y].clone()
    logits_1[u, y] = -float('inf')
    y_others = logits_1.max(dim=-1)[0]

    return y_corr - y_others

def rank_transformation(loss_last, idx_to_fool, loss_history, i_iter):
    n_images = loss_history.shape[0]
    rank = torch.zeros(n_images).to("cuda")
    for i in range(n_images):
        if idx_to_fool[i]:
            if i_iter > 0:
                rank[i] = torch.ge(loss_history[i, :i_iter], loss_last[i]).float().sum() / i_iter
            loss_history[i, i_iter] = loss_last[i]
    return 10 * (rank - 0.5)


def get_square_size(p, image_height, image_width, relaxed=False):
    if relaxed:
        square_size = torch.sqrt(p) * (image_height - 1) + 1
    else:
        square_size = torch.sqrt(p * image_height * image_width)
        square_size = torch.clamp(torch.round(square_size), 1, image_height - 1)

    return square_size


def get_square_size_modified(p, image_height, image_width, relaxed=False):
    square_size = torch.sqrt(p) * (image_height - 1) + 1
    if not relaxed:
        square_size = torch.clamp(torch.round(square_size), 1, image_height - 1)
    return square_size


def odd_int(s):
    """
    returns the largest odd integer that is not larger than s
    for example, odd_int(3.1)=3, odd_int(4)=3, odd_int(5.1)=5
    """
    return 2 * ((s.int() - 1) / 2) + 1


def get_color_vectors(n_channels, eps):
    if n_channels == 1:
        color_vectors = [[-1], [1]]
    elif n_channels == 3:
        color_vectors = torch.tensor([[-1, -1, -1], [-1, -1,  1],
                                      [-1,  1, -1], [-1,  1,  1],
                                      [1, -1, -1], [1, -1,  1],
                                      [1,  1, -1], [1,  1,  1]], dtype=torch.float, device="cuda")
    else:
        raise ValueError("Incorrect number of channels")

    return eps * color_vectors


class MaskHandler:
    def __init__(self):
        self.conv_side = torch.nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=(3, 3),
                                         padding=1)
        self.conv_side.weight = torch.nn.Parameter(torch.tensor([[[
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]]], dtype=torch.float), requires_grad=False)
        self.conv_side.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.conv_side.cuda()

        self.conv_corner = torch.nn.Conv2d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=(3, 3),
                                      padding=1)
        self.conv_corner.weight = torch.nn.Parameter(torch.tensor([[[
            [1, 2, 1],
            [2, -13, 2],
            [1, 2, 1]
        ]]], dtype=torch.float), requires_grad=False)
        self.conv_corner.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.conv_corner.cuda()

    def create_square_masks(self, batch_shape, pos_h, pos_w, s):
        n_images = batch_shape[0]
        image_height = batch_shape[2]
        image_width = batch_shape[3]
        doubled_height = 2 * image_height
        doubled_width = 2 * image_width
        half_height = image_height // 2
        half_width = image_width // 2
        masks_shape = [n_images, 1, doubled_height, doubled_width]
        mask_center_h = doubled_height // 2  # assume resolution dimensions to be even
        mask_center_w = doubled_width // 2

        masks = torch.zeros(masks_shape, dtype=torch.float).cuda()
        masks[:, :, mask_center_h, mask_center_w] = 1

        scaling_matrices = torch.zeros((n_images, 2, 3)).cuda()
        scaling_parameters = 1. / s.float()
        scaling_matrices[:, 0, 0] = scaling_matrices[:, 1, 1] = scaling_parameters
        scaling_grid = affine_grid(scaling_matrices, masks_shape, align_corners=True)
        scaled_masks = grid_sample(masks, scaling_grid, mode="nearest", align_corners=True)

        translation_matrices = torch.zeros((n_images, 2, 3)).cuda()
        translation_matrices[:, 0, 0] = translation_matrices[:, 1, 1] = 1
        unit_h = 2 / (doubled_height - 1)
        unit_w = 2 / (doubled_width - 1)
        steps_h = half_height + pos_h - mask_center_h
        steps_w = half_width + pos_w - mask_center_w
        translation_matrices[:, 0, 2] = -steps_w*unit_w
        translation_matrices[:, 1, 2] = -steps_h*unit_h
        translation_grid = affine_grid(translation_matrices, masks_shape, align_corners=True)
        translated_masks = grid_sample(scaled_masks, translation_grid, mode="nearest", align_corners=True)
        assert all(translated_masks.sum(dim=[1, 2, 3]).cpu() == s.cpu()**2)
        result_masks = translated_masks[:, :, half_height:half_height + image_height,
                       half_width:half_width + image_width]
        assert all(result_masks.sum(dim=[1, 2, 3]).cpu() == s.cpu() ** 2)

        return result_masks

    def create_auxiliary_masks(self, square_masks):
        side_masks = self.conv_side(square_masks.cuda())
        side_masks[side_masks > 1] = 0

        corner_masks = self.conv_corner(square_masks)
        corner_masks[corner_masks != 1] = 0
        #assert all(corner_masks.sum(dim=[1, 2, 3]) == 4)

        return side_masks.bool(), corner_masks.bool()

    def create_all_masks(self, batch_shape, pos_h, pos_w, s):
        square_masks = self.create_square_masks(batch_shape, pos_h, pos_w, s)
        side_masks, corner_masks = self.create_auxiliary_masks(square_masks)
        return square_masks, side_masks, corner_masks

    def format_masks(self, masks, n_channels):
        masks_shape = list(masks.shape)
        masks_shape[1] = n_channels
        return masks.expand(masks_shape).bool()


def softclip(x, min_val, max_val):
    x = min_val + F.softplus(x - min_val, beta=100)  # constraining to x >= min
    x = max_val - F.softplus(max_val - x, beta=100)  # constraining to x <= max
    return x


def parameter_to_numpy(parameter):
    return parameter.data.detach().cpu().numpy()


def plot_color_controller(path, controller):
    d = np.linspace(0.0, 0.5, 16)
    t = np.array([0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])

    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(0, figsize=(16, 12))
    mesh = np.array([(di / controller.improved_rate_init,
                      np.log2(ti/5000+1))
                     for di in d for ti in t], dtype="float32")
    mesh = torch.from_numpy(mesh).cuda()
    output = controller.model(mesh).cpu().detach().numpy()[:, 0]

    plt.imshow(output.reshape((d.shape[0], t.shape[0])).T, vmin=-5, vmax=5)
    plt.xticks(np.arange(d.shape[0]))
    plt.yticks(np.arange(t.shape[0]))
    plt.gca().set_xticklabels(list(map(lambda f: "%.2f" % f, d)))
    plt.gca().set_yticklabels(list(map(lambda f: "%.2f" % f, t)))
    plt.xlabel("Improvement metric")
    plt.ylabel("Iteration")
    plt.colorbar()
    plt.savefig(path + "illustration.png")
    plt.close()


def plot_step_size_controller(path, controller):
    d = np.linspace(0, 0.5, 16)
    t = np.array([0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])

    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(0, figsize=(16, 12))
    mesh = np.array([(di / controller.loss_rate_init,
                      np.log10(ti/5000+1))
                     for di in d for ti in t], dtype="float32")
    mesh = torch.from_numpy(mesh).cuda()
    output = controller.model(mesh)
    output = torch.sigmoid(output)
    output = output.cpu().detach().numpy()[:, 0]

    plt.imshow(output.reshape((d.shape[0], t.shape[0])).T * 31 + 1, vmin=1, vmax=24)
    plt.xticks(np.arange(d.shape[0]))
    plt.yticks(np.arange(t.shape[0]))
    plt.gca().set_xticklabels(list(map(lambda f: "%.2f" % f, d)))
    plt.gca().set_yticklabels(list(map(lambda f: "%.2f" % f, t)))
    plt.xlabel("Improvement metric")
    plt.ylabel("Iteration")
    plt.colorbar()
    plt.savefig(path + "illustration.png")
    plt.close()
