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
        image = cv2.imread(address)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(image)
        _, self.height, self.width = img_tensor.shape
        r = torch.unsqueeze(img_tensor[0], dim=2)
        g = torch.unsqueeze(img_tensor[1], dim=2)
        b = torch.unsqueeze(img_tensor[2], dim=2)
        # Store on CPU; moved to target device on first use
        self.image = torch.cat((r, g, b), dim=2)

    def get_colors(self, points: torch.Tensor) -> torch.Tensor:
        device = points.device
        image = self.image.to(device=device, dtype=points.dtype)

        # UV coords from uv_map are already in [0, 1]
        s_u = torch.clamp(points[:, :, 0], 0.0, 1.0)
        s_v = torch.clamp(points[:, :, 1], 0.0, 1.0)

        ih, iw, _ = image.shape
        i = torch.round((ih - 1) * s_u).long() % ih
        j = torch.round((iw - 1) * s_v).long() % iw

        colors = image[i, j, :]
        return colors
