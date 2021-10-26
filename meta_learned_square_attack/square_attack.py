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


import time
import numpy as np
import torch
import torch.nn.functional as F
from utils import margin_function, get_square_size, get_square_size_modified, odd_int, get_color_vectors
from schedule import LearnedStepSizeController
from logger import Logger

torch.autograd.set_detect_anomaly(False)


def sample_update(height, width, color_vectors, probs,
                    s, x, x_best, delta, mask_handler, temperature=1, update_threshold=0, max_samplings=1,
                    relaxed_squares=False):

    ws = odd_int(s.detach()) if relaxed_squares else s.detach().int()  # window size, include borders if relaxed
    ws_numpy = ws.cpu().numpy()
    if relaxed_squares:
        ws_numpy += 2

    n_channels = x.shape[1]
    n_colors = 2**n_channels
    color_threshold = color_vectors.max() / 2

    n_images = x.shape[0]
    idx_to_update = torch.ones(n_images, dtype=torch.bool, device="cuda")
    result_delta = delta.detach().clone()    # if some perturbation cannot be updated, keep it as is
    result_color_one_hots = -torch.ones(n_images, n_colors, 1, device="cuda")

    samplings_done = 0
    while idx_to_update.sum() > 0 and samplings_done < max_samplings:
        pos_h = torch.tensor(
            np.random.randint(low=0, high=np.maximum(height - ws_numpy + 1, 1)),
            device="cuda",
            dtype=torch.int
        )
        pos_w = torch.tensor(
            np.random.randint(low=0, high=np.maximum(width - ws_numpy + 1, 1)),
            device="cuda",
            dtype=torch.int
        )

        # Hard Gumbel-Softmax sampling from smoothened distribution
        color_one_hots = F.gumbel_softmax(torch.log(probs), tau=temperature, hard=True)
        color = (color_one_hots[:, :, None] * color_vectors[None]).sum(dim=1)[:, :, None, None]

        square_masks = mask_handler.create_square_masks(x.shape, pos_h, pos_w, ws)
        square_masks_f = mask_handler.format_masks(square_masks, n_channels)
        delta_squares = ~square_masks_f * delta + square_masks_f * color
        delta_new = delta_squares

        if relaxed_squares:
            # integer part of a square radius, required for the growth in all directions
            s_frac = torch.clamp((s - ws)/2, min=0)
            s_frac = s_frac[:, None, None, None]
            s_frac_sq = s_frac ** 2

            side_masks, corner_masks = mask_handler.create_auxiliary_masks(square_masks)
            side_masks = mask_handler.format_masks(side_masks, n_channels)
            corner_masks = mask_handler.format_masks(corner_masks, n_channels)
            unchanged_masks = ~(side_masks + corner_masks)

            delta_sides = (1 - s_frac) * delta_squares + s_frac * color
            delta_corners = (1 - s_frac_sq) * delta_squares + s_frac_sq * color
            delta_new = unchanged_masks * delta_squares \
                        + side_masks * delta_sides \
                        + corner_masks * delta_corners

        # compare the color of a square (without sides and corners) with the background
        x_updated = torch.clamp(x.detach() + delta_squares.detach(), 0, 1)
        differences = x_updated - x_best.detach()
        idx_changed = (differences.abs().sum(dim=1) > color_threshold).sum(dim=[1, 2]) > update_threshold * ws * ws
        idx_new = idx_changed & idx_to_update

        result_delta[idx_new] = delta_new[idx_new]
        if samplings_done == 0:
            result_color_one_hots = color_one_hots.unsqueeze(2)
        else:
            result_color_one_hots[idx_new] = color_one_hots[idx_new].unsqueeze(2)
        idx_to_update = ~idx_changed & idx_to_update
        samplings_done += 1

    return result_delta, result_color_one_hots


def square_attack(model, x, y, eps, n_iter, controller, mask_handler,
                  path, metric_labels, log_template, loss_function, temperature=1, update_threshold=0,
                  relaxed_squares=False, verbose=False):
    """
    x - batch of images, y - corresponding labels
    loss_function must have reduction='none' because we operate with losses on the batch images independently
    """

    n_images, n_channels, height, width = x.shape
    color_vectors = get_color_vectors(n_channels, eps)
    logits_clean = model(x)
    margin = margin_function(logits_clean, y)
    logger = Logger(path, n_images, n_iter, n_channels, margin, eps, metric_labels, log_template, verbose)

    corr_classified = torch.eq(logits_clean.argmax(1), y)

    tensors = []
    grad_tensors = []
    if not any(corr_classified):
        # Nothing to be fooled, stop early
        return tensors, grad_tensors, logger.metrics[0]

    x, y = x[corr_classified], y[corr_classified]
    # sample initial vertical line perturbation
    init_delta = torch.tensor(np.random.choice([-eps, eps], size=[x.shape[0], n_channels, 1, width])).to("cuda")
    x_best = torch.clamp(x + init_delta, 0, 1).float()
    x_ae = x_best.detach().clone()

    logits = model(x_best)
    loss_min = loss_function(logits, y).detach()
    margin_min = margin_function(logits, y)
    n_images_to_fool = x.shape[0]
    all_indices = torch.ones(n_images_to_fool, dtype=torch.bool, device="cuda")
    n_queries = np.ones(x.shape[0], dtype=int)
    logger.log_stripes(margin_min)
    n_colors = 2 ** n_channels
    prev_color = torch.zeros(n_images_to_fool, n_colors, 1).to("cuda")
    prev_loss = torch.zeros(n_images_to_fool).to("cuda")
    start_state = {"n_images_to_fool": n_images_to_fool}
    controller.start(start_state)

    logger.start_square_attack(n_images_to_fool=n_images_to_fool)

    meta_loss = 0
    for i_iter in range(n_iter):
        idx_to_fool = margin_min > 0
        if not any(idx_to_fool):
            break

        x_curr, x_best_curr, y_curr = x, x_best, y
        loss_min_curr = loss_min
        margin_min_curr = margin_min
        deltas = (x_best_curr - x_curr).detach()
        state = {"it": i_iter, "prev_color": prev_color.detach(), "prev_loss": prev_loss.detach(), "idx_to_fool": all_indices}
        p, probs = controller.get_parameters(state)
        if type(controller.step_size_controller) is LearnedStepSizeController:
            steps = get_square_size_modified(p, height, width, relaxed=relaxed_squares)
        else:
            steps = get_square_size(p, height, width, relaxed=relaxed_squares)

        deltas, sampled_color_one_hots = sample_update(height, width, color_vectors, probs,
                                          steps, x_curr, x_best_curr, deltas, mask_handler,
                                          temperature=temperature,
                                          update_threshold=update_threshold,
                                          relaxed_squares=relaxed_squares)

        deltas = torch.clamp(deltas, min=-eps, max=eps)
        x_new = torch.clamp(x_curr + deltas, 0, 1)
        logits = model(x_new)
        loss = loss_function(logits, y_curr)
        margin = margin_function(logits, y_curr)

        idx_improved = (loss < loss_min_curr).detach()
        prev_color = sampled_color_one_hots
        prev_loss = idx_improved.float()
        loss_min = (idx_improved * loss + ~idx_improved * loss_min_curr).detach()

        # backprop from current loss term to the input perturbations
        meta_loss_term = torch.clamp(loss - loss_min_curr, max=0).sum()
        if meta_loss_term.requires_grad:
            grad_tensor = torch.autograd.grad(meta_loss_term, deltas)[0]
            tensors.append(deltas)
            grad_tensors.append(grad_tensor)
        meta_loss += meta_loss_term.detach()

        margin_min = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = torch.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool.cpu().numpy()] += 1
        x_ae[idx_to_fool] = x_best[idx_to_fool]
        # log the performance metrics
        logger.log_squares(i_iter, margin_min, n_queries, time.time(),
                           prev_loss, all_indices, steps, meta_loss,
                           probs, sampled_color_one_hots)

    if (x_ae - x).abs().max() >= eps + 1e-7:
        print("Warning: Perturbation beyond bounds, will be clipped.")
        x_ae = x + torch.clamp(x_ae - x, -eps, eps)
    if x_ae.max() >= 1. + 1e-7 or x_ae.min() <= -1e-7:
        print("Warning: Perturbation beyond bounds, will be clipped.")
        x_ae = torch.clamp(x_ae, 0, 1)

    last_iter = int(np.max(n_queries)) - 2
    logger.finish_square_attack(x, y, x_ae - x, n_queries, last_iter)
    return tensors, grad_tensors, logger.metrics[last_iter]
