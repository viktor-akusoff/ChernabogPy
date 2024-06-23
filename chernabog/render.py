"""Describes RayTracer classes for rendering an image."""

import tqdm  # type: ignore
import torch
from .entities import Scene
from .misc import Utils as u
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
import asyncio


class RayTraceStrategy:

    def calc_steps(
        self,
        points: torch.Tensor,
        velocity: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return points, velocity


class LinearRayTrace(RayTraceStrategy):

    def calc_steps(
        self,
        points: torch.Tensor,
        velocity: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v = velocity
        v_norm = v / torch.unsqueeze(torch.linalg.norm(v, dim=2), dim=2)
        new_velocity = v_norm * eps
        return (
            points + new_velocity * torch.unsqueeze(mask, dim=2),
            new_velocity
        )


class SchwarzschildRayTrace(RayTraceStrategy):

    def __init__(self, curvature: float = 3) -> None:
        self.curvature = curvature

    def calc_steps(
        self,
        points: torch.Tensor,
        velocity: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        square_areal_velocity = torch.pow(
            torch.einsum('ijk,ijk->ij', points, velocity), 2
        )
        points = points + velocity * eps * torch.unsqueeze(mask, dim=2)
        accel = (
            points *
            (
                -abs(self.curvature) * square_areal_velocity /
                torch.pow(
                    torch.einsum('...i,...i', points, points),
                    2.5
                )
            )[:, :, None,]
        )
        velocity = u.m_norm(velocity + accel * eps)

        return points, velocity


class ReissnerNordstromRayTrace(RayTraceStrategy):

    def __init__(self, e=1.5, charge: float = 1) -> None:
        self.e = e
        eps_0 = 8.854
        G = 6.674
        charge_radius = (charge ** 2 * G * 10) / (4 * torch.pi * eps_0)
        self.A = charge_radius

    def calc_steps(
        self,
        points: torch.Tensor,
        velocity: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-3
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        A = self.A
        e = self.e

        r = torch.linalg.norm(points)
        r2 = torch.einsum('...i,...i', points, points)
        r3 = torch.pow(r2, 1.5)

        square_areal_velocity = torch.pow(
            torch.einsum('ijk,ijk->ij', points, velocity), 2
        )

        r_factor = 1 - 1 / r

        square_areal_velocity = torch.pow(
            torch.einsum('ijk,ijk->ij', points, velocity), 2
        )
        points = points + velocity * eps * torch.unsqueeze(mask, dim=2)
        c = (
            1.5 / r2
            - 2 * A / r3
            + A * e**2 *
            (r_factor + A / r2) / r_factor *
            (1 / r) * (2 - 1 / r) / r_factor ** 2
        )
        accel = (
            points *
            (
                -c * square_areal_velocity /
                r3
            )[:, :, None,]
        )
        velocity = u.m_norm(velocity + accel * eps)

        return points, velocity


class RayTracer:

    def __init__(
        self,
        scene: Scene,
        strategy: RayTraceStrategy,
        eps: float = 1e-3
    ) -> None:
        super().__init__()
        self.scene = scene
        self.strategy = strategy
        self.eps = eps
        self._progress: float = 0
        self._rays: torch.Tensor | None = None

    async def async_calc_points(self):
        rays = self.scene.camera.rays
        velocity = rays - self.scene.camera.position_m
        sdf = self.scene.sdf_matrix(rays)
        bool_mask = torch.abs(sdf) > self.eps
        int_mask = bool_mask.long()
        total = sdf[bool_mask].shape[0]
        present, old_present = 0, 0
        while sdf[bool_mask].shape[0]:
            old_present = present
            present = round(100 - (sdf[bool_mask].shape[0] / total) * 100)
            rays, velocity = self.strategy.calc_steps(
                rays,
                velocity,
                int_mask,
                self.eps
            )
            sdf = self.scene.sdf_matrix(rays)
            bool_mask = torch.abs(sdf) > self.eps
            int_mask = bool_mask.long()
            self._progress = present - old_present
            await asyncio.sleep(0)
        self._progress = -1
        self._rays = rays
        return rays

    @property
    def points(self):
        if self._rays is None:
            asyncio.run(self.async_calc_points())
        return self._rays

    async def async_show_progress(self):
        with tqdm.tqdm(total=100) as bar:
            while self._progress >= 0:
                bar.update(self._progress)
                await asyncio.sleep(0)
            bar.close()

    def calc_image(self):
        return self.scene.get_colors(self.points).cpu().numpy()

    def view_3d_rays_hits(
        self,
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        z_lim: Tuple[float, float],
        w_lim: int = 1,
        h_lim: int = 1
    ):
        w = self.scene.camera._w
        h = self.scene.camera._h
        w_step = w // w_lim
        h_step = h // h_lim
        points = self.points[::h_step, ::w_step]
        new_h, new_w, _ = points.shape
        data = points.reshape(new_w * new_h, 3).cpu().numpy()
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir='z', s=1)
        ax.set_title("Rays' hits 3d graph")
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim3d(x_lim)
        ax.set_ylim3d(y_lim)
        ax.set_zlim3d(z_lim)
        plt.show()
