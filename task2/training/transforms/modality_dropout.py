import numpy as np
from gardening_tools.modules.transforms.BaseTransform import BaseTransform


class Torch_ModalityDropout(BaseTransform):
    """Zeroes one or more entire modality channels with probability
    `p_per_sample`, to reduce the model's learned dependency on those
    specific channels -- unlike gardening_tools' per-(sample,channel)
    intensity transforms (independent Bernoulli draw per channel),
    modality dropout here is gated ONCE per sample and applies to the
    whole listed channel set together, matching how "a modality is
    missing/unreliable" actually manifests (not independently per
    channel).

    User request 2026-08-21 (Task 2, channel_order=[flair, dwi_b1000,
    swi_or_t2s]): default channels=(2,) targets ONLY the 3rd modality
    (swi_or_t2s) specifically, motivated by known per-subject quality
    issues in that channel (e.g. sub-20's t2s misregistration, found
    earlier this session via a pairwise-mask-Dice screen) -- a model that
    has learned to lean on this channel is more exposed to exactly that
    kind of failure. Complements Torch_Cutout3D (cutout.py): Cutout
    regularizes spatial over-reliance, this regularizes channel
    over-reliance.

    pixel_value=0.0 matches this pipeline's background convention
    (asparagus_volume_wise_znorm forces background to 0.0), so a dropped
    channel looks like "this channel's foreground is entirely
    background", not an out-of-distribution constant."""

    def __init__(
        self,
        data_key: str = "image",
        p_per_sample: float = 0.2,
        channels: tuple = (2,),
        pixel_value: float = 0.0,
        batched: bool = True,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.channels = list(channels)
        self.pixel_value = pixel_value
        self.batched = batched

    def __drop__(self, image):
        # image: (C, D, H, W)
        for c in self.channels:
            image[c] = self.pixel_value
        return image

    def __call__(self, data_dict):
        if not self.batched:
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key] = self.__drop__(data_dict[self.data_key])
        else:
            for b in range(data_dict[self.data_key].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b] = self.__drop__(data_dict[self.data_key][b])
        return data_dict
