"""
This script demonstrates
 * the multi-shell sampling pattern for diffusion MRI.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""


import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sigpy.mri import dvs

import os

fontsize1 = 12
fontsize2 = 16

# %% Import and Plot Diffusion Vector Sets (DVS)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')

dvs_files = ['Internal_20.dvs', 'Internal_30.dvs', 'Internal_64.dvs']
colors = ['g', 'b', 'r'] # plt.cm.ocean(np.linspace(1, 0.4, len(dvs_files)))

for n in range(len(dvs_files)):

    instr = DIR_PATH + '/' + dvs_files[n]

    # read dvs file
    print(str(n) + '. read dvs file ' + instr)
    directions, CoordinateSystem, Normalisation, Vector = dvs.read(instr)

    # scale the diffusion vector
    # b_ui = 3000
    # match directions:
    #     case 20:
    #         b_actual = 1000
    #     case 30:
    #         b_actual = 2000
    #     case 64:
    #         b_actual = 3000

    b_ui = 3000
    if directions == 20:
        b_actual = 1000
    elif directions == 30:
        b_actual = 2000
    elif directions == 64:
        b_actual = 3000

    Vector_radius = np.sqrt(b_actual / b_ui)
    print('   radius ' + str(Vector_radius))

    # plot surface
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = np.sin(phi) * np.cos(theta) * Vector_radius
    y = np.sin(phi) * np.sin(theta) * Vector_radius
    z = np.cos(phi) * Vector_radius

    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    color='grey', alpha=0.1, linewidth=0)

    # plot samples on the surface
    xx = Vector[:, 0] * Vector_radius
    yy = Vector[:, 1] * Vector_radius
    zz = Vector[:, 2] * Vector_radius

    ax.scatter(xx, yy, zz, color=colors[n], s=26, marker='*',
               label='$b = $' + f'{b_actual:4}' + '$~$s$/$mm$^2$')

ax.legend(loc='upper left', fontsize=fontsize1)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

ax.set_xticks([-1, 1]) #, fontsize=fontsize2)
ax.set_yticks([-1, 1]) #, fontsize=fontsize2)
ax.set_zticks([-1, 1]) #, fontsize=fontsize2)

ax.set_xlabel('$g_x$') #, fontsize=fontsize2)
ax.set_ylabel('$g_y$') #, fontsize=fontsize2)
ax.set_zlabel('$g_z$') #, fontsize=fontsize2)

plt.tight_layout()
plt.savefig(DIR_PATH + '/' + 'spheres.png',
            bbox_inches='tight', pad_inches=0, dpi=300)