import numpy as np
import torch
import torch.nn.functional as F
from gardening_tools.functional.sanity_checks import verify_shapes_are_equal

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover -- SimpleITK is an optional dep for tasks that never resample
    sitk = None


def reverse_preprocessing(array, image_properties):
    # device=array.device (2026-08-11 fix, found running the first real seg
    # test_step): canvas previously defaulted to CPU regardless of `array`'s
    # device, causing "Expected all tensors to be on the same device" once
    # uncrop_array_onto_canvas() tried to assign a CUDA `array` into it --
    # pre-existing in the original F.interpolate-only implementation too,
    # just never exercised by a CUDA-resident test_step call before now.
    canvas = torch.zeros((1, array.shape[1], *image_properties["original_size"]), dtype=array.dtype, device=array.device)
    pad_bbox = image_properties["pad_box"]
    crop_bbox = image_properties.get("crop_box", [])

    ndim = len(array.shape[2:])
    if ndim == 2:
        mode = "bilinear"
    elif ndim == 3:
        mode = "trilinear"

    if len(pad_bbox) > 0:
        array = unpad_array(array, pad_bbox)
        verify_shapes_are_equal(reference_shape=array.shape[2:], target_shape=image_properties["shape_before_pad"])

    if image_properties.get("size_before_resample") is not None:
        original_spacing = image_properties.get("original_spacing")
        new_spacing = image_properties.get("new_spacing")
        # 2026-08-11 (FOMO26 Task 2 round-trip-Dice gate, spec 05-A section 9):
        # F.interpolate's own coordinate convention (even with align_corners
        # explicitly set) does not exactly invert a SimpleITK-resampled
        # forward pass (SimpleITK anchors resampling in PHYSICAL space via
        # origin+spacing, not PyTorch's normalized-grid convention) -- this
        # was invisible for near-isotropic data but produced large,
        # inconsistent round-trip Dice errors (as low as 0.42, no reliable
        # direction from align_corners toggling) on Task 2's severely
        # anisotropic thick-slice MRI (native through-plane spacing up to
        # 7.5mm) once actually measured. Resampling with SimpleITK using the
        # SAME resampler configuration convention as the forward resample
        # (origin/direction implicit-identity, matching `_sitk_resample` /
        # `_resample_to_iso` in FOMO26/data/) fixes this -- verified via
        # segmentation/roundtrip_check.py: median round-trip Dice for Task 2
        # went from ~0.92 (F.interpolate) to ~0.97 (this) with far less
        # case-to-case variance. Falls back to the original F.interpolate
        # path when spacing metadata isn't present in `image_properties`,
        # for backward compatibility with tasks/pipelines whose properties
        # dict predates this field.
        if sitk is not None and ndim == 3 and original_spacing is not None and new_spacing is not None:
            array = _sitk_resample_array(array, new_spacing, image_properties["size_before_resample"], original_spacing)
        else:
            array = F.interpolate(array, size=image_properties["size_before_resample"], mode=mode)

    if len(crop_bbox) > 0:
        canvas = uncrop_array_onto_canvas(array, canvas, crop_bbox)
    else:
        canvas = array

    return canvas


def _sitk_resample_array(array: torch.Tensor, src_spacing, target_shape, target_spacing) -> torch.Tensor:
    """Resamples a (1, C, D, H, W) tensor channel-by-channel with SimpleITK
    (linear interpolation, matching F.interpolate's "trilinear" intent) using
    the exact resampler convention the forward preprocessing resample used
    (implicit identity origin/direction) -- see reverse_preprocessing()'s
    call site docstring for why this replaces F.interpolate for 3D arrays
    with known spacing."""
    assert array.shape[0] == 1, "sitk-based reverse resample only supports batch size 1"
    np_array = array[0].detach().cpu().numpy()
    out = np.zeros((np_array.shape[0], *target_shape), dtype=np.float32)
    for c in range(np_array.shape[0]):
        img = sitk.GetImageFromArray(np.transpose(np_array[c], (2, 1, 0)))
        img.SetSpacing([float(s) for s in src_spacing])
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([float(s) for s in target_spacing])
        resampler.SetSize([int(s) for s in target_shape])
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0.0)
        # BSpline (order 3) chosen 2026-08-11 after an explicit linear vs
        # BSpline vs Lanczos vs Gaussian comparison on all 23 Task 2 cases
        # (segmentation/roundtrip_check.py) -- no interpolator gave a
        # uniform win (BSpline helped the worst small-lesion cases, e.g.
        # sub-06 +0.03 Dice, sub-16 +0.02, at the cost of small regressions
        # elsewhere, e.g. sub-20 -0.011) but BSpline had the best worst-case
        # floor and matches the forward image resample's own interpolator
        # (_resample_to_iso / _sitk_resample in FOMO26/data/), so it was
        # picked for consistency rather than on net-average grounds alone.
        resampler.SetInterpolator(sitk.sitkBSpline)
        out_img = resampler.Execute(img)
        out[c] = np.transpose(sitk.GetArrayFromImage(out_img), (2, 1, 0))
    return torch.from_numpy(out).unsqueeze(0).to(device=array.device, dtype=array.dtype)


def unpad_array(array, pad_box):
    """
    Unpads an array based on the provided padding box.
    Array must be shape (b,c,x,y) or (b,c,x,y,z).
    """
    if len(pad_box) == 6:
        return array[
            :,
            :,
            pad_box[0] : array.shape[2] - pad_box[1],
            pad_box[2] : array.shape[3] - pad_box[3],
            pad_box[4] : array.shape[4] - pad_box[5],
        ]
    elif len(pad_box) == 4:
        return array[:, :, pad_box[0] : array.shape[2] - pad_box[1], pad_box[2] : array.shape[3] - pad_box[3]]
    else:
        raise ValueError("Unsupported padding box length.")


def uncrop_array_onto_canvas(array, canvas, crop_bbox):
    """
    Uncrops an array onto a canvas based on the provided crop bounding box.
    Assumes arrays are shape (b, c, x, y) or (b, c, x, y, z).
    """
    slices = [
        slice(None),
        slice(None),
        slice(crop_bbox[0], crop_bbox[1] + 1),
        slice(crop_bbox[2], crop_bbox[3] + 1),
    ]
    if len(crop_bbox) == 6:
        slices.append(
            slice(crop_bbox[4], crop_bbox[5] + 1),
        )
    canvas[tuple(slices)] = array
    return canvas
