# Copyright 2022 The Procter & Gamble Company
#
# Licenced under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the original License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" Data generator following Lorenz-I equations.
    Used to generate chaotic time series used in Kennel1992 and Frank2001.
"""

import matplotlib.pyplot as plt
import numpy as np
import random

from scipy.integrate import RK45


plt.style.use("tableau-colorblind10")


def lorenz_system(_, yvec, r=45.92, b=4, sigma=16):
    ret = np.zeros_like(yvec)
    x, y, z = yvec
    ret[0] = sigma * (y - x)
    ret[1] = -1 * x * z + r * x - y
    ret[2] = x * y - b * z
    return ret


def generate_1d_lorenz(n_steps, coord='x', yvec0=[0, 0.1, 1,], noise_pct=0, tmstep=0.01):
    idx = { 'x': 0, 'y': 1, 'z': 2, }[coord]
    integral = RK45(
            fun=lorenz_system,
            t0=0, y0=yvec0,
            t_bound=n_steps,
            first_step=tmstep, max_step=tmstep
    )
    data = []
    for _ in range(n_steps):
        integral.step()
        yt = integral.y[idx]
        if noise_pct > 0:
            ns = noise_pct / 100
            yt += random.uniform((1 - ns) * yt, (1 + ns) * yt)

        data.append(yt)

    data = np.reshape(data, (-1, 1))
    return data


if __name__ == "__main__":
    print(lorenz_system(0, [0., 0.1, 1]))
    n_pts = 2000
    data = generate_1d_lorenz(n_pts)
    noisy = generate_1d_lorenz(n_pts, noise_pct=10)
    _, ax = plt.subplots(dpi=200)
    ax.plot(range(n_pts), data, label='clean')
    ax.plot(range(n_pts), noisy, label='noisy')
    ax.legend(loc="upper right")
    plt.show()

