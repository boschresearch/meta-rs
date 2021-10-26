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


import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden, final_bias=True):
        super(MLP, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.a = nn.ModuleList([nn.ReLU()])

        self.n_hidden = n_hidden
        for i in range(n_hidden - 1):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
            self.a.append(nn.ReLU())

        self.fc.append(nn.Linear(hidden_size, output_size, bias=final_bias))

    def forward(self, x):
        output = self.fc[0](x)
        output = self.a[0](output)

        for i in range(self.n_hidden - 1):
            output = self.fc[i + 1](output)
            output = self.a[i + 1](output)

        output = self.fc[self.n_hidden](output)
        return output
