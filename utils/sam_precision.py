"""Precision helpers for SAM predictors used by evaluation tasks."""

from __future__ import annotations

import torch


def _patch_prompt_encoder_coord_dtype(predictor) -> None:
    pe_layer = getattr(getattr(predictor.model, "prompt_encoder", None), "pe_layer", None)
    if pe_layer is None or getattr(pe_layer, "_sparsesam_dtype_patch", False):
        return

    def _matrix_dtype_device() -> tuple[torch.dtype, torch.device]:
        matrix = pe_layer.positional_encoding_gaussian_matrix
        return matrix.dtype, matrix.device

    def forward_model_dtype(size: tuple[int, int]) -> torch.Tensor:
        dtype, device = _matrix_dtype_device()
        height, width = size
        grid = torch.ones((height, width), device=device, dtype=dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / height
        x_embed = x_embed / width
        pe = pe_layer._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords_model_dtype(coords_input: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        dtype, device = _matrix_dtype_device()
        coords = coords_input.clone().to(device=device)
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return pe_layer._pe_encoding(coords.to(dtype=dtype))

    pe_layer.forward = forward_model_dtype
    pe_layer.forward_with_coords = forward_with_coords_model_dtype
    pe_layer._sparsesam_dtype_patch = True


def _to_dtype(value, dtype: torch.dtype):
    if torch.is_tensor(value) and torch.is_floating_point(value):
        return value.to(dtype=dtype)
    if isinstance(value, list):
        return [_to_dtype(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_dtype(item, dtype) for item in value)
    if isinstance(value, dict):
        return {key: _to_dtype(item, dtype) for key, item in value.items()}
    return value


def _patch_mask_decoder_input_dtype(predictor) -> None:
    decoder = getattr(predictor.model, "mask_decoder", None)
    if decoder is None or getattr(decoder, "_sparsesam_dtype_patch", False):
        return

    original_forward = decoder.forward

    def forward_with_decoder_dtype(*args, **kwargs):
        decoder_dtype = next(decoder.parameters()).dtype
        args = tuple(_to_dtype(arg, decoder_dtype) for arg in args)
        kwargs = {key: _to_dtype(value, decoder_dtype) for key, value in kwargs.items()}
        return original_forward(*args, **kwargs)

    decoder.forward = forward_with_decoder_dtype
    decoder._sparsesam_dtype_patch = True


def enable_sam_dtype_for_predictor(
    predictor,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Run the full SAM predictor model in `dtype`.

    SAM-HQ's predictor preprocesses input images in fp32 by default. This helper
    converts the full SAM model, then wraps `predictor.set_torch_image` so the
    image encoder receives inputs in the same dtype as the model and stores
    encoded features in the mask decoder dtype.
    """
    predictor.model.to(dtype=dtype)
    _patch_prompt_encoder_coord_dtype(predictor)
    _patch_mask_decoder_input_dtype(predictor)
    encoder_dtype = next(predictor.model.image_encoder.parameters()).dtype
    decoder_dtype = next(predictor.model.mask_decoder.parameters()).dtype

    @torch.no_grad()
    def set_torch_image_with_model_dtype(transformed_image: torch.Tensor, original_image_size):
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == predictor.model.image_encoder.img_size
        ), "set_torch_image input must be BCHW with long side equal to image_encoder.img_size."

        predictor.reset_image()
        predictor.original_size = original_image_size
        predictor.input_size = tuple(transformed_image.shape[-2:])
        input_image = predictor.model.preprocess(transformed_image).to(dtype=encoder_dtype)
        features, interm_features = predictor.model.image_encoder(input_image)
        predictor.features = features.to(dtype=decoder_dtype)
        predictor.interm_features = [
            feat.to(dtype=decoder_dtype) if torch.is_floating_point(feat) else feat
            for feat in interm_features
        ]
        predictor.is_image_set = True

    predictor.set_torch_image = set_torch_image_with_model_dtype


def enable_image_encoder_dtype_for_predictor(
    predictor,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Backward-compatible alias for the full-model dtype path."""
    enable_sam_dtype_for_predictor(predictor, dtype=dtype)
