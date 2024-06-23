"""Different objects with their SDF matrices."""
import torch
import numpy as np
from .misc import Utils as U
from .misc import CachingFields
from typing import Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
from .textures import BaseTexture


class Camera(CachingFields):
    """Class of an observer from which position image will be created."""

    def __init__(
        self,
        position: torch.Tensor,
        look_at: torch.Tensor,
        resolution: Tuple[int, int],
        distance_to_screen: float,
        fov: float
    ) -> None:

        super().__init__()
        self.position: torch.Tensor = position
        self.look_at: torch.Tensor = look_at
        self.distance_to_screen: float = distance_to_screen
        self.fov: float = fov
        self._w, self._h = resolution

    @property
    @CachingFields.cache_field('_screen_center')
    def screen_center(self) -> torch.Tensor:
        """Center of a screen on which image will be projected."""

        result = (
            (self.look_at - self.position) /
            torch.linalg.norm(self.look_at - self.position)
        ) * self.distance_to_screen

        return result

    @property
    @CachingFields.cache_field('_up')
    def up(self) -> torch.Tensor:
        """Center of a screen on which image will be projected."""

        axe = torch.tensor(
            [0, 0, -1],
            dtype=self.position.dtype,
            device=self.position.device
        )

        side = torch.linalg.cross(self.screen_center, axe)

        result = torch.linalg.cross(self.screen_center, side)

        return result

    @property
    @CachingFields.cache_field('_basis_vectors')
    def basis_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Basis vectors of a screen's pixel grid."""

        screen_center_cross_up = torch.linalg.cross(
            self.screen_center,
            self.up
        )

        v: torch.Tensor = (
            screen_center_cross_up /
            torch.linalg.norm(screen_center_cross_up)
        ) * self.pixel_size

        v_cross_screen_center = torch.linalg.cross(
            v,
            self.screen_center
        )

        u: torch.Tensor = (
            v_cross_screen_center /
            torch.linalg.norm(v_cross_screen_center)
        ) * self.pixel_size

        return (u, v)

    @property
    @CachingFields.cache_field('_pixel_size')
    def pixel_size(self) -> float:
        """Size of a single pixel of a projection screen."""

        pixel_size = (
            2 * np.tan(self.fov / 2) *
            self.distance_to_screen / self._h
        )

        return pixel_size

    @property
    @CachingFields.cache_field('_start')
    def starting_point(self) -> torch.Tensor:
        """Top right point of a projection screen."""
        S = self.screen_center
        u, v = self.basis_vectors
        w, h = self._w, self._h
        start = S + (h / 2) * u + (w / 2) * v - u/2 - v/2
        return start

    @property
    @CachingFields.cache_field('_position_m')
    def position_m(self):
        return U.v_repeat(self.position, self._w, self._h)

    @property
    @CachingFields.cache_field('_start_m')
    def start_m(self):
        return U.v_repeat(self.starting_point, self._w, self._h)

    @property
    @CachingFields.cache_field('_rays')
    def rays(self):
        """Matrix of starting rays casted by a camera"""

        u, v = self.basis_vectors
        u_m = U.v_repeat(u, self._w, self._h)
        v_m = U.v_repeat(v, self._w, self._h)

        start_m = self.start_m
        position_m = self.position_m

        hor_m = torch.arange(
            0, self._w, 1,
            dtype=u.dtype,
            device=u.device

        ).repeat([self._h, 1])

        ver_m = torch.arange(
            0, self._h, 1,
            dtype=u.dtype,
            device=u.device
        ).unsqueeze(dim=1).repeat([1, self._w])

        ub_m = torch.einsum('ijk, ij->ijk', u_m, ver_m)
        vb_m = torch.einsum('ijk, ij->ijk', v_m, hor_m)

        screen_points = start_m - ub_m - vb_m

        rays = position_m + screen_points

        return rays

    @property
    @CachingFields.cache_field('_verts')
    def verts(self) -> List:
        v = torch.cat(
            (
                self.position.unsqueeze(dim=-2),
                self.rays[0][0].unsqueeze(dim=-2),
                self.rays[0][-1].unsqueeze(dim=-2),
                self.rays[-1][0].unsqueeze(dim=-2),
                self.rays[-1][-1].unsqueeze(dim=-2),
            ),
            dim=-2
        ).cpu().numpy()
        verts = [
            [v[0], v[1], v[2]],
            [v[0], v[3], v[4]],
            [v[0], v[2], v[4]],
            [v[0], v[3], v[1]],
            [v[1], v[2], v[4], v[3]],
        ]
        return verts

    def draw_3d(
        self,
        ax: Axes3D
    ) -> None:
        """Shows a camera and a projection screen positions in a 3d space."""

        ax.add_collection3d(
            Poly3DCollection(
                self.verts,
                facecolors='cyan',
                linewidths=1,
                edgecolors='b',
                alpha=.25
            )
        )
        ax.quiver(
            *self.position.cpu().numpy(),
            *self.screen_center.cpu().numpy()*2
        )
        ax.quiver(
            *self.position.cpu().numpy(),
            *(self.up).cpu().numpy()*1.5
        )


class BaseEntity:
    """Base class for every entity on scene."""

    def __init__(
        self,
        position: torch.Tensor
    ):

        self.position = position
        self._position_m: torch.Tensor | None = None

    def position_diff(self, matrix_of_points: torch.Tensor) -> torch.Tensor:
        if (
            self._position_m is None or
            self._position_m.shape != matrix_of_points.shape
        ):
            h, w, _ = matrix_of_points.shape
            self._position_m = U.v_repeat(self.position, w, h)
        return matrix_of_points - self._position_m

    def move(self, direction: torch.Tensor):
        """Moves entity by direction."""
        self.position += direction

    def sdf_matrix(self, matrix_of_points: torch.Tensor) -> torch.Tensor:
        """Returns matrix of distances (m, n, 1) from points of
        matrix of points (m, n, 3) to entity."""
        return torch.abs(matrix_of_points - self.position)

    def draw_3d(self, ax: Axes3D) -> None:
        pass

    def uv_map(self, matrix_of_points: torch.Tensor) -> torch.Tensor:
        return matrix_of_points


class Sphere(BaseEntity):

    def __init__(
        self,
        position: torch.Tensor,
        radius: float,
        inverse: bool = False
    ):
        super().__init__(position)
        self.radius = radius
        self.inverse: bool = inverse

    def sdf_matrix(self, matrix_of_points: torch.Tensor) -> torch.Tensor:

        sign = -1 if self.inverse else 1

        return sign * (
            torch.linalg.norm(
                self.position_diff(matrix_of_points),
                dim=2
            ) - self.radius
        )

    def draw_3d(self, ax: Axes3D) -> None:

        if self.inverse:
            return

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x1, y1, z1 = self.position.cpu().numpy()

        x = self.radius * np.outer(np.cos(u), np.sin(v)) + x1
        y = self.radius * np.outer(np.sin(u), np.sin(v)) + y1
        z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z1

        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='g')

    def uv_map(self, matrix_of_points: torch.Tensor):
        p_diff = self.position_diff(matrix_of_points)
        p_x = p_diff[:, :, 0]
        p_y = p_diff[:, :, 1]
        p_z = p_diff[:, :, 2]
        p_r = torch.linalg.norm(p_diff, dim=2)
        p_theta = torch.acos(p_z / p_r)
        p_phi = torch.arctan2(p_x, p_y)
        phi_a_bool = p_phi < 0
        phi_b_bool = p_phi >= 0
        phi_a_int = phi_a_bool.long()
        phi_b_int = phi_b_bool.long()
        phi_a = (p_phi + torch.pi) * phi_a_int
        phi_b = (p_phi - torch.pi) * phi_b_int
        p_phi = phi_a + phi_b
        u_p_theta = torch.unsqueeze(p_theta, dim=2)
        u_p_phi = torch.unsqueeze(p_phi, dim=2)
        return torch.cat((u_p_theta, u_p_phi), dim=2)


class FlatRing(BaseEntity):

    def __init__(
        self,
        position: torch.Tensor,
        radiuses: Tuple[float, float],
        norm_vector: torch.Tensor
    ):
        super().__init__(position)
        self.inner_radius, self.outer_radius = radiuses
        self.norm_vector = norm_vector

    def sdf_matrix(self, matrix_of_points: torch.Tensor) -> torch.Tensor:

        distance_vectors = U.proj_on_plane(
            self.position_diff(matrix_of_points),
            self.norm_vector
        )
        distances = torch.norm(distance_vectors, dim=2)

        over = torch.logical_and(
            (distances >= self.inner_radius),
            (distances <= self.outer_radius)
        ).long() * torch.abs(
            torch.norm(matrix_of_points - distance_vectors, dim=2)
        )

        distance_vectors_norm = torch.nn.functional.normalize(
            distance_vectors,
            p=2,
            dim=2
        )

        inner_points = distance_vectors_norm * self.inner_radius
        below_m = distances < self.inner_radius
        below = below_m.long() * torch.abs(
            torch.norm(matrix_of_points - inner_points, dim=2)
        )

        outer_points = distance_vectors_norm * self.outer_radius
        outer_m = distances > self.outer_radius
        upper = outer_m.long() * torch.abs(
            torch.norm(matrix_of_points - outer_points, dim=2)
        )

        return over + below + upper

    def draw_3d(self, ax: Axes3D) -> None:
        pass

    def uv_map(self, matrix_of_points: torch.Tensor):
        p_diff = self.position_diff(matrix_of_points)
        p_x = p_diff[:, :, 0]
        p_y = p_diff[:, :, 1]
        p_r = (
            (torch.norm(matrix_of_points, dim=2) - self.inner_radius) %
            (self.outer_radius - self.inner_radius)
        ) / (self.outer_radius - self.inner_radius)
        p_phi = torch.arctan2(p_x, p_y)

        p_phi_mask_bool = p_phi < 0
        p_phi_mask_int = p_phi_mask_bool.long()
        p_phi += 2 * p_phi_mask_int * torch.pi
        p_phi /= 2 * torch.pi

        u_p_r = torch.unsqueeze(p_r, dim=2)
        u_p_phi = torch.unsqueeze(p_phi, dim=2)

        return torch.cat((u_p_r, u_p_phi), dim=2)


class DecoratedPair:

    def __init__(self, entity: BaseEntity, texture: BaseTexture) -> None:
        self.entity = entity
        self.texture = texture

    def sdf_matrix(self, matrix_of_points: torch.Tensor) -> torch.Tensor:
        return self.entity.sdf_matrix(matrix_of_points)

    def render(
        self,
        matrix_of_points: torch.Tensor,
        eps: float = 1e-3
    ) -> torch.Tensor:
        sdf_matrix = self.sdf_matrix(matrix_of_points)
        abs_sdf = torch.abs(sdf_matrix)
        calc_region = abs_sdf <= eps
        uv_map = self.entity.uv_map(matrix_of_points)
        colors = self.texture.get_colors(uv_map)
        return colors * torch.unsqueeze(calc_region, dim=2)


class Scene:
    """Describes scene to render."""
    def __init__(
        self,
        camera: Camera
    ):
        self.camera: Camera = camera
        self.pairs: List[DecoratedPair] = []

    def push_entity(self, pair: DecoratedPair):
        """Pushes new entity to the list."""
        self.pairs.append(pair)

    def pop_entity(self) -> DecoratedPair:
        """Pops out last entity from the list."""
        pair: DecoratedPair = self.pairs[-1]
        del self.pairs[-1]
        return pair

    def view_3d_scene(
        self,
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        z_lim: Tuple[float, float]
    ):

        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection='3d')

        for pair in self.pairs:
            pair.entity.draw_3d(ax)

        self.camera.draw_3d(ax)

        ax.set_xlim3d(x_lim)
        ax.set_ylim3d(y_lim)
        ax.set_zlim3d(z_lim)
        ax.set_title("3d scene")
        ax.set_box_aspect((1, 1, 1))
        plt.show()

    def sdf_matrix(self, matrix_of_points: torch.Tensor):
        result = self.pairs[0].entity.sdf_matrix(matrix_of_points)
        for pair in self.pairs[1:]:
            result = torch.min(
                result,
                pair.entity.sdf_matrix(matrix_of_points)
            )
        return result

    def get_colors(self, matrix_of_points: torch.Tensor, eps: float = 1e-3):
        result = torch.zeros_like(
            matrix_of_points,
            device=matrix_of_points.device,
            dtype=matrix_of_points.dtype
        )

        for pair in self.pairs:
            sdf = pair.entity.sdf_matrix(matrix_of_points)
            bool_mask = sdf <= eps
            int_mask = torch.unsqueeze(bool_mask.long(), dim=2)
            uv = pair.entity.uv_map(matrix_of_points)
            colors = pair.texture.get_colors(uv) * int_mask
            result += colors

        return result
