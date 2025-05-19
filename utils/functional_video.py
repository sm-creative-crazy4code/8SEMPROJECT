import torch


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError(f"clip should be Tesnor. Got {type(clip)}")

    if not clip.ndimension() == 4:
        raise ValueError(f"clip should be 4D. Got {clip.dim()}D")

    return True


def crop(clip, i, j, h, w):
   
    assert len(clip.size()) == 4, "clip should be a 4D tensor"
    return clip[..., i : i + h, j : j + w]


def resize(clip, target_size, interpolation_mode):
    assert len(target_size) == 2, "target size should be tuple (height, width)"
    # print(target_size)
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )


def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
  
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip


def center_crop(clip, crop_size):
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    assert h >= th and w >= tw, "height and width must be no smaller than crop_size"

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def to_tensor(clip):
   
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError(
            f"clip tensor should have data type uint8. Got {str(clip.dtype)}"
        )
    return clip.float().permute(3, 0, 1, 2) / 255.0


def normalize(clip, mean, std, inplace=False):
    
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


def hflip(clip):
   
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    return clip.flip(-1)
