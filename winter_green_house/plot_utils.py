import astropy.units as au
import numpy as np
import vedo as vp
from vedo.applications import RayCastPlotter

from winter_green_house.astropy_utils import quantity_to_np


def plot_3d_volume(volume: au.Quantity, x: au.Quantity, y: au.Quantity, z: au.Quantity):
    volume = quantity_to_np(volume)

    x = quantity_to_np(x)
    y = quantity_to_np(y)
    z = quantity_to_np(z)
    spacing = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
    origin = [x[0], y[0], z[0]]
    vol = vp.Volume(volume, spacing=spacing, origin=origin)

    # vol.add_scalarbar3d(title='Volume')
    # vol.cmap('viridis')
    # print(f"Volume: {volume}")
    #
    # lego = vol.legosurface(vmin=volume.min(), vmax=volume.max())
    # lego.cmap('viridis').add_scalarbar3d()
    ray_cast = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7)

    # text1 = vp.Text2D("Volume", pos="bottom-left", c='k')
    # text2 = vp.Text2D("Volume", pos="top-right", c='k')
    text3 = vp.Text2D("Volume", pos="top-right", c='k')

    vp.show([(ray_cast, text3)], N=1, azimuth=10, elevation=0).close()

    # plt.show(viewup="z")
    # plt.close()


def test_plot_3d_volume():
    x = np.linspace(-1, 1, 10) * au.m
    y = np.linspace(-1, 1, 10) * au.m
    z = np.linspace(-1, 1, 10) * au.m
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    volume = (X ** 2 + Y ** 2 + Z ** 2) * au.m ** (-2)
    plot_3d_volume(volume, x=x, y=y, z=z)
