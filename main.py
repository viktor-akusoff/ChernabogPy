import torch
import numpy as np
import asyncio
from chernabog.misc import TensorFabric
from chernabog.entities import Sphere, FlatRing, Scene, Camera, DecoratedPair
from chernabog.textures import ColorTexture, CheckersTexture, ImageTexture
from chernabog.render import LinearRayTrace, RayTracer, SchwarzschildRayTrace, ReissnerNordstromRayTrace
import matplotlib.pyplot as plt

w = 640
h = 320


async def main(raytracer: RayTracer):
    async with asyncio.TaskGroup() as tg:
        tg.create_task(raytracer.async_calc_points())
        tg.create_task(raytracer.async_show_progress())

if __name__ == "__main__":

    t = TensorFabric(torch.float64)

    black_hole_texture = CheckersTexture((0, 0, 0.9), (0.5, 0.5, 0), 15)
    background_texture = CheckersTexture((0, 0.2, 0.7), (0.3, 0.9, 0), 25)
    ring_texture = CheckersTexture((0.9, 0, 0), (0, 0.5, 0.5), 20)

    black_hole = Sphere(t([0, 0, 0]), 0.5)
    background = Sphere(t([0, 0, 0]), 9, inverse=True)
    ring = FlatRing(t([0, 0, 0]), (0.5, 5), t([0, 0, 1]))

    camera = Camera(t([-7, 0, 1.2]), t([0, 0, 0]), (w, h), 1, np.pi/3)

    scene = Scene(camera)

    pair_black_hole = DecoratedPair(black_hole, black_hole_texture)
    pair_background = DecoratedPair(background, background_texture)
    pair_ring = DecoratedPair(ring, ring_texture)

    scene.push_entity(pair_black_hole)
    scene.push_entity(pair_background)
    scene.push_entity(pair_ring)

    raytracer = RayTracer(scene, ReissnerNordstromRayTrace(1, 0.5), eps=1e-3)

    asyncio.run(main(raytracer))

    # raytracer.view_3d_rays_hits(
    #     (-15, 15),
    #     (-15, 15),
    #     (-15, 15),
    #     64,
    #     32
    # )

    img_data = raytracer.calc_image()

    plt.imsave("./blackholeHD.png", img_data)
    plt.show()