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
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import File, save_progress_plot


class Logger:
    def __init__(self, path, n_images, n_iter, n_channels, margin, eps, metric_labels, log_template, verbose):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.n_images = n_images
        self.n_iter = n_iter
        self.n_colors = 2 ** n_channels
        self.verbose = verbose

        # these values will be defined after the stripe initialization
        self.n_images_to_fool = -1
        self.attack_metrics = None
        self.color_probs = None
        self.color_sampled = None
        self.image_log = []
        self.color_prob_log = []
        self.color_sampled_log = []
        self.time_start = 0

        self.n_correct_init = (margin > 0).sum().detach().cpu().numpy()
        clean_acc = self.n_correct_init / n_images
        self.batch_log = File(self.path + "batch_log.txt")
        self.batch_log.write('Clean accuracy: {:.2%}, # images: {}, eps={:.2}'.
                             format(clean_acc, n_images, eps))

        self.metric_labels = metric_labels
        self.log_template = log_template
        self.n_metrics = len(metric_labels)
        self.metrics = np.zeros((n_iter, self.n_metrics))
        self.checkpoints = [500, 1000, 2000, 3000, 4000, 5000]
        self.margin_min = None

    def log_stripes(self, margin):
        self.margin_min = margin.detach().cpu().numpy()
        n_correct = (margin > 0).sum().detach().cpu().numpy()
        acc = n_correct / self.n_images
        stripe_metrics = [0, acc, 1, 1, margin.mean(), 0]
        self.batch_log.write('stripes: ' + self.log_template.format(*stripe_metrics))

    def start_square_attack(self, n_images_to_fool):
        self.n_images_to_fool = n_images_to_fool

        if self.verbose:
            for i in range(n_images_to_fool):
                folder = self.path + "images/image_{}/".format(i)
                os.makedirs(folder)
                self.image_log.append(File(folder + "image_log.txt"))
                self.color_prob_log.append(File(folder + "color_prob_log.txt"))
                self.color_sampled_log.append(File(folder + "color_sampled_log.txt"))

            self.attack_metrics = -np.ones((n_images_to_fool, self.n_iter, 2))
            self.color_probs = np.empty((n_images_to_fool, self.n_iter, self.n_colors))
            self.color_sampled = np.empty((n_images_to_fool, self.n_iter, self.n_colors))


        self.time_start = time.time()

    def log_squares(self, i_iter, margin_tensor, n_queries, time_iter,
                    loss_last, idx_to_fool, steps, meta_loss,
                    color_distribution, color_sampled):
        margin = margin_tensor.detach().cpu().numpy()
        self.margin_min = np.minimum(self.margin_min, margin)
        n_correct = (self.margin_min > 0.0).sum()
        acc = n_correct / self.n_images
        idx_ae = self.margin_min <= 0
        exist_ae = any(idx_ae)
        mean_nq_ae = np.mean(n_queries[idx_ae]) if exist_ae else 0
        median_nq_ae = np.median(n_queries[idx_ae]) if exist_ae else 0
        avg_margin_min = np.mean(self.margin_min)

        meta_loss_np = meta_loss.detach().cpu().numpy()
        self.metrics[i_iter] = [meta_loss_np, acc, mean_nq_ae, median_nq_ae, avg_margin_min, time_iter-self.time_start]
        self.batch_log.write('{}: '.format(i_iter) + self.log_template.format(*self.metrics[i_iter]))
#        if i_iter + 1 in self.checkpoints:
#            print('{}: '.format(i_iter + 1) + self.log_template.format(*self.metrics[i_iter]))

        # register the image-level metrics
        if self.verbose:
            self.__log_image_level(i_iter, idx_to_fool, loss_last, steps, color_distribution, color_sampled)

    def __log_image_level(self, i_iter, idx_to_fool_tensor, loss_last, steps,
                          color_distribution, color_sampled):
        idx_to_fool = idx_to_fool_tensor.detach().cpu().numpy()
        self.attack_metrics[idx_to_fool, i_iter, 0] = loss_last.detach().cpu().numpy()
        self.attack_metrics[idx_to_fool, i_iter, 1] = steps.detach().cpu().numpy()
        self.color_probs[idx_to_fool, i_iter, :] = color_distribution.detach().cpu().numpy()
        self.color_sampled[idx_to_fool, i_iter] = color_sampled.detach().cpu().numpy()[:, :, 0]
        for i in range(self.n_images_to_fool):
            if idx_to_fool[i]:
                self.image_log[i].write('{}: prev_loss={:.6f} step={:.2f}'.
                                        format(i_iter + 1, *self.attack_metrics[i, i_iter]))
                self.color_prob_log[i].write(
                    '%06d: %s' % (i_iter + 1,
                                  self.color_probs[i, i_iter]))
                self.color_sampled_log[i].write(
                    '%06d: %s' % (i_iter + 1,
                                  self.color_sampled[i, i_iter]))

    def finish_square_attack(self, images, labels, deltas, n_queries, last_iter):

        cifar10_labels = ["0_airplane", "1_automobile", "2_bird", "3_cat", "4_deer",
                          "5_dog", "6_frog", "7_horse", "8_ship", "9_truck"]
        np.save(self.path + "metrics.npy", self.metrics)
        for i_metric in range(self.n_metrics):
            metric_name = self.metric_labels[i_metric]
            filepath = self.path + metric_name + ".png"
            save_progress_plot(filepath, last_iter, self.metrics[None, :, i_metric], [metric_name])

        if self.verbose:
            colors = ["k", "b", "g", "c", "r", "m", "y", "gray"] if self.n_colors == 8 else ["k", "w"]
            for i in range(self.n_images_to_fool):
                image_path = self.path + "images/image_{}/".format(i)
                n_iter = int(n_queries[i]) - 1
                save_progress_plot(image_path + "step_size.png", n_iter, self.attack_metrics[i].transpose(),
                                   labels=["loss", "step"])

                # fig, ax = plt.subplots()
                # bar_labels = np.arange(last_iter+1)
                # for i_color in range(self.n_colors):
                #     ax.bar(bar_labels, self.color_probs[i, :, i_color],
                #            bottom=self.color_probs[i, :, :i_color].sum(axis=1),
                #            color=colors[i_color],
                #            width=1)
                #
                # fig.savefig(image_path + "color.png")
                # plt.close(fig)

                image = images[i].detach().cpu().numpy().transpose((1, 2, 0))
                delta = deltas[i].detach().cpu().numpy().transpose((1, 2, 0))
                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(image)
                #axs[0].set_title("{}".format(cifar10_labels[labels[i].detach().cpu().numpy()]))
                axs[1].imshow(5 * delta + 0.5)
                axs[1].set_title("# queries: {}".format(n_iter))
                axs[2].imshow(np.clip(image + delta, 0, 1))
                axs[2].set_title("perturbed image")
                fig.savefig(image_path + "image.pdf")
                plt.close(fig)


