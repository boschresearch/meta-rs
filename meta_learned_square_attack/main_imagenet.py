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
import shutil
import argparse
import pickle
import numpy as np
import time
__import__("matplotlib").use("TkAgg")

import torch
from schedule import ConstantStepSize, FixedStepSize, Controller, \
    ConstantColor, LearnedColorController, LearnedStepSizeController
from utils import BatchedModel, save_progress_plot, File, MaskHandler, \
    margin_function, minus_ce, plot_color_controller, plot_step_size_controller
from square_attack import square_attack
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import math

torch.autograd.set_detect_anomaly(False)

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--datapath')
parser.add_argument('--model', default="resnet50")
parser.add_argument("--model_dir")
parser.add_argument('--eps', default=4 / 255)
parser.add_argument('--n_images', default=128, type=int,
                    help="number of the dataset images to work with (taken from the beginning)")
parser.add_argument('--n_iter', type=int, default=500)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--bs', default=-1, type=int,
                    help="batch size (set to -1, if all images should be attacked at once)")
parser.add_argument('--meta_optimizer', type=str, default="adam", help="adam | sgd")
parser.add_argument('--meta_lr', type=float, default=0.05)
parser.add_argument('--meta_schedule', type=str, default="constant", help="constant | cosine")
parser.add_argument('--baseline', type=str, default="none", help="none | meta_learned_square_attack | auto_attack")
parser.add_argument('--p', type=float, default=0.8)
parser.add_argument('--color', default="uniform", type=str,
                    help="uniform | mlp | path to a stored controller")
parser.add_argument('--step_size', default="fixed_truncated", type=str,
                    help="constant | fixed_rescaled | fixed_truncated | mlp | path to a stored controller")
parser.add_argument('--n_hidden', default=2, type=int, help="number of hidden layers")
parser.add_argument('--hidden_size', default=10, type=int, help="size of a hidden layers")
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--min_prob', type=float, default=0.05)
parser.add_argument('--rate_init', type=float, default=0.25)
parser.add_argument('--update_threshold', type=float, default=0)
parser.add_argument('--loss', type=str, default="ce", help="ce | margin")
parser.add_argument('--test', action='store_true')
parser.add_argument('--verbose', action='store_true', help="flag to log image-level information")
parser.add_argument('--relaxed_squares', action='store_true')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--output', default="output/")
parser.add_argument('--seed', default=None, type=int)


def main():

    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.random.manual_seed(args.seed)

    dataset = ImageNet(args.datapath)
    _, val_loader = dataset.make_loaders(batch_size=args.bs, workers=0,
                                         only_val=True, shuffle_val=True)

    model, _ = make_and_restore_model(arch=args.model, dataset=dataset,
                                      resume_path=args.model_dir)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of {args.model} parameters: {n_params}")

    for param in model.parameters():
        param.requires_grad = False

    metric_labels = ["meta_loss", "acc", "avg#q_ae", "med#q_ae", "avg_margin", "time"]
    log_template = "meta_loss={:.4f} acc={:.2%} avg#q_ae={:.2f} med#q_ae={:.1f} avg_margin={:.2f} time={:.2f}s"
    n_metrics = len(metric_labels)
    batch_metrics = np.zeros((int(math.ceil(args.n_images / args.bs)), n_metrics))
    epoch_metrics = np.zeros((args.n_epochs, n_metrics))
    training_log = File(args.output + "training_log.txt")

    if os.path.exists(args.output):
        shutil.rmtree(args.output)           # delete previous results
    os.makedirs(args.output)

    torch.set_grad_enabled(False)

    n_channels = 3
    n_colors = 2 ** n_channels
    color_distribution_init = torch.ones(n_colors, device="cuda") / n_colors

    model = BatchedModel(model, batch_size=128, num_classes=1000)

    if args.baseline == "meta_learned_square_attack":
        args.color = "uniform"
        args.step_size = "fixed_rescaled"
        args.p = 0.3
        args.loss = "margin"
    elif args.baseline == "auto_attack":
        args.color = "uniform"
        args.step_size = "fixed_truncated"
        args.p = 0.8
        args.loss = "margin"
    elif args.baseline != "none":
        raise ValueError("Unknown baseline")

    if args.color == "uniform":
        color_controller = ConstantColor(color_distribution_init)
    elif args.color == "mlp":
        color_controller = LearnedColorController(n_colors=n_colors,
                                                  n_hidden=args.n_hidden,
                                                  hidden_size=args.hidden_size,
                                                  momentum=args.momentum,
                                                  min_prob=args.min_prob,
                                                  improved_rate_init=args.rate_init)
    elif os.path.exists(args.color):
        with open(args.color, 'rb') as f:
            color_controller = pickle.load(f)
        color_controller.n_iter = args.n_iter
    else:
        raise ValueError("Unknown controller type")

    if args.step_size == "constant":
        step_size_controller = ConstantStepSize(p_init=args.p)
    elif args.step_size == "fixed_rescaled" or args.step_size == "fixed_truncated":
        rescale_schedule = args.step_size == "fixed_rescaled"
        step_size_controller = FixedStepSize(p_init=args.p,
                                             n_iter=args.n_iter,
                                             rescale_schedule=rescale_schedule)
    elif args.step_size == "mlp":
        step_size_controller = LearnedStepSizeController(n_hidden=args.n_hidden,
                                                         hidden_size=args.hidden_size,
                                                         momentum=args.momentum,
                                                         loss_rate_init=args.rate_init)
    elif os.path.exists(args.step_size):
        with open(args.step_size, 'rb') as f:
            step_size_controller = pickle.load(f)
        step_size_controller.n_iter = args.n_iter
    else:
        raise ValueError("Unknown controller type")

    controller = Controller(step_size_controller, color_controller)

    mask_handler = MaskHandler()
    if args.loss == "ce":
        loss_function = minus_ce
    elif args.loss == "margin":
        loss_function = margin_function
    else:
        raise ValueError("Unknown loss function type")

    for i_e in range(args.n_epochs):
        print("\nEpoch {}\n".format(i_e))

        epoch_path = args.output + "training/epoch_{}/".format(i_e)
        epoch_log = File(epoch_path + "epoch_log.txt")
        for i_b, (images, labels) in enumerate(val_loader):
            if i_b * args.bs >= args.n_images:
                break

            batch_path = epoch_path + "batch_{}/".format(i_b)
            tensors, grad_tensors, metrics = square_attack(model,
                                   images.cuda(),
                                   labels.cuda(),
                                   args.eps, args.n_iter, controller, mask_handler,
                                   batch_path, metric_labels, log_template,
                                   loss_function=loss_function,
                                   temperature=args.temperature,
                                   update_threshold=args.update_threshold,
                                   relaxed_squares=args.relaxed_squares,
                                   verbose=args.verbose)
            batch_metrics[i_b] = metrics
            print('batch {}: '.format(i_b) + log_template.format(*batch_metrics[i_b]))
            epoch_log.write('batch {}: '.format(i_b) + log_template.format(*batch_metrics[i_b]))
            for i_metric in range(n_metrics):
                metric_name = metric_labels[i_metric]
                filepath = epoch_path + metric_name + ".png"
                save_progress_plot(filepath, i_b + 1, batch_metrics[None, :, i_metric], [metric_name])

            if (i_b + 1) * args.bs >= args.n_images:
                break

        if args.color != "uniform":
            plot_color_controller(epoch_path + "color/", controller.color_controller)
        if args.step_size not in ["constant", "fixed_rescaled", "fixed_truncated"]:
            plot_step_size_controller(epoch_path + "step_size/", controller.step_size_controller)

        epoch_metrics[i_e] = np.mean(batch_metrics, axis=0)
        print('\nepoch {}: '.format(i_e) + log_template.format(*epoch_metrics[i_e]))
        training_log.write('epoch {}: '.format(i_e) + log_template.format(*epoch_metrics[i_e]))

        for i_metric in range(n_metrics):
            metric_name = metric_labels[i_metric]
            filepath = args.output + metric_name + ".png"
            save_progress_plot(filepath, i_e + 1, epoch_metrics[None, :, i_metric], [metric_name])


if __name__ == '__main__':
    main()
