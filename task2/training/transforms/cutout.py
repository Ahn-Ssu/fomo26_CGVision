import numpy as np
from gardening_tools.modules.transforms.BaseTransform import BaseTransform


class Torch_Cutout3D(BaseTransform):
    """3D Cutout (DeVries & Taylor 2017, adapted to 3D volumes): zeroes one
    or more random cuboid sub-regions of the volume, using the SAME region
    across all channels -- unlike gardening_tools' per-(sample,channel)-
    gated intensity transforms (Torch_Blur etc.), Cutout removes a spatial
    region of information from the joint sample, so every channel must
    lose the same region together (otherwise the network could just read
    the missing region from another channel, defeating the point).

    User request 2026-08-21: reduce the model's learned dependency on
    specific spatial regions / any single modality by combining this with
    Torch_ModalityDropout (modality_dropout.py) -- Cutout targets spatial
    over-reliance, ModalityDropout targets channel over-reliance,
    complementary regularizers for the same underlying goal.

    `size_range` is a fraction of EACH spatial dim (independently sampled
    per axis, so cuboids aren't forced to be isotropic) -- 0.0 corresponds
    to "same background value everywhere already", following this
    pipeline's convention (asparagus_volume_wise_znorm forces background
    to 0.0, so pixel_value=0.0 matches erased regions to the existing
    background value rather than introducing an out-of-distribution
    constant)."""

    def __init__(
        self,
        data_key: str = "image",
        p_per_sample: float = 0.15,
        size_range: tuple = (0.1, 0.3),
        pixel_value: float = 0.0,
        n_cuts: int = 1,
        batched: bool = True,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.size_range = size_range
        self.pixel_value = pixel_value
        self.n_cuts = n_cuts
        self.batched = batched

    def __cutout__(self, image):
        # image: (C, D, H, W) -- same cuboid slice applied across all C
        spatial_shape = image.shape[1:]
        for _ in range(self.n_cuts):
            slices = [slice(None)]
            for dim_size in spatial_shape:
                frac = float(np.random.uniform(*self.size_range))
                cut_size = max(1, int(round(dim_size * frac)))
                cut_size = min(cut_size, dim_size)
                start = int(np.random.randint(0, dim_size - cut_size + 1))
                slices.append(slice(start, start + cut_size))
            image[tuple(slices)] = self.pixel_value
        return image

    def __call__(self, data_dict):
        if not self.batched:
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key] = self.__cutout__(data_dict[self.data_key])
        else:
            for b in range(data_dict[self.data_key].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b] = self.__cutout__(data_dict[self.data_key][b])
        return data_dict
