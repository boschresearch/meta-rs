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


from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from basic import MLP


class StepSizeController(ABC):

    @abstractmethod
    def start(self, state):
        """set parameters for starting the attack"""

    @abstractmethod
    def get_trainable_parameters(self):
        """return parameters to optimize"""

    @abstractmethod
    def get_step_size(self, state):
        """returns size of the next step"""


class ColorController(ABC):

    @abstractmethod
    def start(self, state):
        """set parameters for starting the attack"""

    @abstractmethod
    def get_trainable_parameters(self):
        """return parameters to optimize"""

    @abstractmethod
    def get_color_logits(self, state):
        """returns logits of the color distribution"""


class Controller:

    def __init__(self, step_size_controller,
                 color_controller):
        self.step_size_controller = step_size_controller
        self.color_controller = color_controller

    def start(self, state):
        self.step_size_controller.start(state)
        self.color_controller.start(state)

    def get_trainable_parameters(self):
        for param in self.step_size_controller.get_trainable_parameters():
            yield param
        for param in self.color_controller.get_trainable_parameters():
            yield param

    def get_parameters(self, state):
        return self.step_size_controller.get_step_size(state), \
               self.color_controller.get_color_logits(state)


class ConstantStepSize(StepSizeController):

    def __init__(self, p_init):
        self.p = p_init

    def start(self, state):
        pass

    def get_trainable_parameters(self):
        return []

    def get_step_size(self, state):
        """ Constant schedule """
        n_images = state["idx_to_fool"].int().sum()
        return torch.full((n_images,), self.p).cuda()


class FixedStepSize(StepSizeController):

    def __init__(self, p_init, n_iter, rescale_schedule=False):
        # assume image is square and get a fraction of modified pixels
        self.p_init = p_init
        self.n_iter = n_iter
        self.rescale_schedule = rescale_schedule

    def start(self, state):
        pass

    def get_trainable_parameters(self):
        return []

    def get_step_size(self, state):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = state["it"]
        if self.rescale_schedule:
            it = int(it / self.n_iter * 10000)
        n_images = state["idx_to_fool"].int().sum()

        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it <= 10000:
            p = self.p_init / 512
        else:
            p = self.p_init

        return torch.full((n_images,), p).cuda()


class ConstantColor(ColorController):

    def __init__(self, color_logits_init):
        self.color_logits = color_logits_init

    def start(self, state):
        pass

    def get_trainable_parameters(self):
        return []

    def get_color_logits(self, state):
        """ Constant schedule """
        n_images = state["idx_to_fool"].int().sum()
        return self.color_logits.repeat((n_images, 1))


class LearnedColorController(ColorController):
    def __init__(self, n_colors=8,
                 hidden_size=10, n_hidden=2, momentum=0.99,
                 min_prob=0.05, improved_rate_init=0.25,
                 model=None):
        self.momentum = momentum
        self.min_prob = min_prob
        self.improved_rate_init = improved_rate_init
        self.n_colors = n_colors
        self.input_size = 2
        self.output_size = 1

        if model is not None:
            self.model = model
        else:
            self.model = MLP(self.input_size, self.output_size,
                             hidden_size=hidden_size, n_hidden=n_hidden,
                             final_bias=True).to("cuda")

    def get_trainable_parameters(self):
        return self.model.parameters()

    def start(self, state):
        self.n_images_to_fool = state["n_images_to_fool"]
        self.color_improved_rate = torch.full((self.n_images_to_fool, self.n_colors),
                                              self.improved_rate_init).cuda()

    def get_color_logits(self, state):
        it = state["it"]
        prev_color = state["prev_color"].squeeze(2)
        prev_loss = state["prev_loss"].unsqueeze(1)

        update = prev_color * prev_loss + (1 - prev_color) * self.color_improved_rate
        self.color_improved_rate = self.momentum * self.color_improved_rate + (1 - self.momentum) * update

        model_input = torch.stack((self.color_improved_rate / self.improved_rate_init,
                                   torch.log2(torch.full_like(self.color_improved_rate, it/5000 + 1))),
                                  dim=2).detach()

        color_logits = self.model(model_input.reshape((-1, self.input_size)))
        color_logits = color_logits.reshape(self.color_improved_rate.shape)

        probs = F.softmax(color_logits, dim=1)
        probs = (1 - self.min_prob * self.n_colors) * probs + self.min_prob

        return probs


class LearnedStepSizeController(StepSizeController):
    def __init__(self, momentum=0.99, loss_rate_init=0.25,
                 hidden_size=10, n_hidden=2, model=None):
        self.momentum = momentum
        self.loss_rate_init = loss_rate_init

        if model is not None:
            self.model = model
        else:
            self.model = MLP(input_size=2, output_size=1, hidden_size=hidden_size, n_hidden=n_hidden).to("cuda")

    def start(self, state):
        self.n_images_to_fool = state["n_images_to_fool"]
        self.loss_rate = torch.full((self.n_images_to_fool, 1), self.loss_rate_init).cuda()

    def get_trainable_parameters(self):
        return self.model.parameters()

    def get_step_size(self, state):
        it = state["it"]
        prev_loss = state["prev_loss"].unsqueeze(1)

        self.loss_rate = self.momentum * self.loss_rate + (1 - self.momentum) * prev_loss
        model_input = torch.cat((self.loss_rate / self.loss_rate_init,
                                 torch.log2(torch.full((self.n_images_to_fool, 1), it/5000 + 1, device="cuda"))),
                                dim=1).detach()
        output = self.model(model_input).view(-1)
        p_current = torch.sigmoid(output)

        return p_current**2
