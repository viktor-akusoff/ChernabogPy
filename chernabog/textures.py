"""Everything you need for texture mapping."""

import cv2
import torch
import torchvision.transforms as transforms  # type: ignore
from typing import Tuple
from PIL import Image as im
from .misc import Utils as U


class BaseTexture:
    """Class describes base class for textures."""

    def __init__(self):
        pass

    def get_colors(self, points: torch.Tensor):
        """Returns color of point."""


class ColorTexture(BaseTexture):
    """Class describing single color texture."""
    def __init__(self, color: Tuple[float, float, float]):
        self.color = color

    def get_colors(self, points: torch.Tensor) -> torch.Tensor:
        color = torch.tensor(
            self.color,
            dtype=points.dtype,
            device=points.device
        )
        w, h, _ = points.shape
        result = U.v_repeat(color, h, w)
        return result


class CheckersTexture(BaseTexture):
    """Class describing checkers color texture."""
    def __init__(
        self,
        color1: Tuple[float, float, float],
        color2: Tuple[float, float, float],
        length: float
    ):
        self.color1 = color1
        self.color2 = color2
        self.length = length

    def get_colors(self, points: torch.Tensor) -> torch.Tensor:
        w, h, _ = points.shape

        color1 = torch.tensor(
            self.color1,
            dtype=points.dtype,
            device=points.device
        )
        color1_m = U.v_repeat(color1, h, w)

        color2 = torch.tensor(
            self.color2,
            dtype=points.dtype,
            device=points.device
        )
        color2_m = U.v_repeat(color2, h, w)

        s_uv = points * 100 % (2 * self.length)

        color1_mask_bool = torch.logical_and(
            ((s_uv[:, :, 1] // self.length) == 0),
            ((s_uv[:, :, 0] // self.length) != 0)
        ) + torch.logical_and(
            ((s_uv[:, :, 1] // self.length) != 0),
            ((s_uv[:, :, 0] // self.length) == 0)
        )
        color2_mask_bool = torch.logical_not(color1_mask_bool)

        color1_mask_int = torch.unsqueeze(color1_mask_bool.long(), dim=2)
        color2_mask_int = torch.unsqueeze(color2_mask_bool.long(), dim=2)

        return color1_m * color1_mask_int + color2_m * color2_mask_int


class ImageTexture(BaseTexture):
    """Class describing texture made from external image."""

    def __init__(
        self,
        address
    ):
        super().__init__()
        self.image = im.open(address)
        image = cv2.imread(address)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.image = transform(image)
        _, self.height, self.width = self.image.shape
        r = torch.unsqueeze(self.image[0], dim=2)
        g = torch.unsqueeze(self.image[1], dim=2)
        b = torch.unsqueeze(self.image[2], dim=2)

        self.image = torch.cat((r, g, b), dim=2)

    def get_colors(self, points: torch.Tensor) -> torch.Tensor:
        s_u = points[:, :, 0] / torch.max(points)
        s_v = points[:, :, 1] / torch.max(points)

        i = torch.unsqueeze(torch.round((self.height - 1) * s_u), dim=2)
        j = torch.unsqueeze(torch.round((self.width - 1) * s_v), dim=2)

        ind = torch.cat((i, j), dim=2).cpu()

        h, w, _ = ind.shape

        ih, iw, _ = self.image.shape

        ind[ind < -ih] = torch.abs(ind[ind < -ih] + ih)
        
        colors = torch.zeros(h, w, 3)

        colors[:, :, 0] = self.image[
            ind[:, :, 0].long(), ind[:, :, 1].long(),
            0
        ]
        colors[:, :, 1] = self.image[
            ind[:, :, 0].long(), ind[:, :, 1].long(),
            1
        ]
        colors[:, :, 2] = self.image[
            ind[:, :, 0].long(), ind[:, :, 1].long(),
            2
        ]

        return colors.cuda()
