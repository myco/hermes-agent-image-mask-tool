#!/usr/bin/env python3
"""Mask-guided object insertion via the ChatGPT/Codex Responses API (three-image variant).

Same fixed contract as ``tools/codex_image_mask.py`` (``gpt-image-2`` at ``quality: high``,
``action: "edit"``, ``input_fidelity: "high"`` with the same backend fallback) but takes THREE
images: the main scene, the mask, and an *object reference* image whose subject is to be placed
inside the masked region. All shared plumbing (auth, image loading/validation, SSE streaming,
retries) is imported from ``tools.codex_image_mask``.

Payload shape by ``CODEX_IMAGE_MASK_MODE``:

* ``tool`` (default) — content parts: ``input_image`` main, ``input_image`` object; the mask goes
  to the hosted tool's ``input_image_mask`` (documented Responses shape). The host model therefore
  sees only TWO attached images.
* ``inline`` — content parts: ``input_image`` main, ``input_image`` mask, ``input_image`` object;
  no ``input_image_mask``. The host model sees THREE images.

Because the visible image count differs per mode, the tool prepends a deterministic role
preamble ("Attached images, in order: (1) ... (2) ...") to the user's prompt so numbered
references like "Image 3 contains the object" never point at the wrong picture. The preamble
actually used is echoed back in the result as ``image_roles``.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Optional

from tools.codex_image_mask import (
    IMAGE_ACTION, IMAGE_MODEL, IMAGE_QUALITY, IMAGE_TIER, INPUT_FIDELITY, PROVIDER,
    _CODEX_BASE_URL, _CODEX_CHAT_MODEL, _NO_AUTH, _NONFINAL_RETRIES, _PARTIAL_IMAGES_REQUESTED,
    _collect_image_b64, _httpx_available, _load_image_bytes, _mask_mode, _png_pixel_size,
    _read_codex_access_token, _rejects_input_fidelity, _to_data_url, _validate_pair,
)
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

TOOL_NAME = "codex_image_mask_with_reference"

_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image editing requests by using the "
    "image_generation tool when provided. The user supplies a main scene image, a mask that "
    "defines the only editable region, and a reference image containing an object. Extract "
    "the object from the reference image and place it inside the masked region of the main "
    "scene, preserving the object's exact details and keeping everything outside the mask "
    "unchanged.")

# Role preambles keyed by mask mode: which attached image is which, in the order the host
# model receives them (the mask is invisible to the host model in ``tool`` mode).
_IMAGE_ROLES = {
    "tool": (
        "Attached images, in order: (1) the main scene to edit; (2) the reference image "
        "containing the object to insert. The editable region is defined by the provided "
        "image mask (transparent pixels); everything outside it must stay unchanged."),
    "inline": (
        "Attached images, in order: (1) the main scene to edit; (2) the mask — its "
        "transparent pixels mark the ONLY region you may repaint, opaque pixels must stay "
        "unchanged; (3) the reference image containing the object to insert."),
}


def check_codex_image_mask_with_reference_requirements() -> bool:
    return bool(_read_codex_access_token()) and _httpx_available()


def _build_responses_payload(
    *, prompt: str, main_data_url: str, mask_data_url: str, object_data_url: str, mode: str,
) -> Dict[str, Any]:
    """Responses body for one masked object-insertion edit (see module docstring for the
    per-mode part order). No ``tool_choice``: the Codex backend rejects every forcing shape."""
    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": f"{_IMAGE_ROLES[mode]}\n\n{prompt}"},
        {"type": "input_image", "image_url": main_data_url},
    ]
    tool: Dict[str, Any] = {
        "type": "image_generation",
        "model": IMAGE_MODEL,
        "action": IMAGE_ACTION,
        "input_fidelity": INPUT_FIDELITY,
        "quality": IMAGE_QUALITY,
        "size": "auto",
        "output_format": "png",
        "partial_images": _PARTIAL_IMAGES_REQUESTED,
    }
    if mode == "inline":
        content.append({"type": "input_image", "image_url": mask_data_url})
    else:
        tool["input_image_mask"] = {"image_url": mask_data_url}
    content.append({"type": "input_image", "image_url": object_data_url})
    return {
        "model": _CODEX_CHAT_MODEL,
        "store": False,
        "instructions": _CODEX_INSTRUCTIONS,
        "input": [{"type": "message", "role": "user", "content": content}],
        "tools": [tool],
        "stream": True,
    }


def _error(error: str, error_type: str, prompt: str = "", **extra: Any) -> Dict[str, Any]:
    return {"success": False, "image": None, "error": error, "error_type": error_type,
            "model": IMAGE_TIER, "prompt": prompt, "provider": PROVIDER, **extra}


def codex_image_mask_with_reference_tool(
    prompt: str, image_url: str, mask_url: str, object_url: str,
) -> Dict[str, Any]:
    """Place the object from ``object_url`` inside the ``mask_url`` region of ``image_url``
    according to ``prompt``. Returns the same ``{"success", "image", ...}`` dict shape as
    ``image_generate`` / ``codex_image_mask``."""
    prompt = (prompt or "").strip()
    if not prompt:
        return _error("Prompt is required and must be a non-empty string", "invalid_argument")
    token = _read_codex_access_token()
    if not token:
        return _error(_NO_AUTH, "auth_required", prompt)
    if not _httpx_available():
        return _error("httpx Python package not installed (pip install httpx)", "missing_dependency", prompt)

    try:
        main_raw = _load_image_bytes(image_url, "image_url (main image)")
        mask_raw = _load_image_bytes(mask_url, "mask_url (mask image)")
        object_raw = _load_image_bytes(object_url, "object_url (object reference image)")
        validation = _validate_pair(main_raw, mask_raw)
        main_data_url = _to_data_url(main_raw, "image_url (main image)")
        mask_data_url = _to_data_url(mask_raw, "mask_url (mask image)")
        object_data_url = _to_data_url(object_raw, "object_url (object reference image)")
    except Exception as exc:  # noqa: BLE001
        return _error(f"Invalid image input for masked object insertion: {exc}", "invalid_image_input", prompt)

    mode = _mask_mode()
    payload = _build_responses_payload(
        prompt=prompt, main_data_url=main_data_url, mask_data_url=mask_data_url,
        object_data_url=object_data_url, mode=mode)
    attempts = _NONFINAL_RETRIES + 1
    input_fidelity: str = INPUT_FIDELITY
    logger.info("Masked object insertion via Codex (%s, quality=%s, mask_mode=%s) — prompt: %s",
                IMAGE_MODEL, IMAGE_QUALITY, mode, prompt[:80])
    try:
        collected: Optional[Dict[str, str]] = None
        for attempt in range(attempts):
            try:
                collected = _collect_image_b64(token, payload)
            except RuntimeError as exc:
                # Same backend quirk as codex_image_mask: drop only ``input_fidelity`` and retry.
                if _rejects_input_fidelity(exc) and "input_fidelity" in payload["tools"][0]:
                    logger.warning("Codex backend rejected input_fidelity (%s); retrying without it.", exc)
                    payload["tools"][0].pop("input_fidelity")
                    input_fidelity = "unsupported_by_backend"
                    collected = _collect_image_b64(token, payload)
                else:
                    raise
            if collected and collected.get("source") == "final" and collected.get("b64"):
                break
            if attempt < _NONFINAL_RETRIES:
                logger.warning("Codex image stream ended without a final result (attempt %s/%s); retrying.",
                               attempt + 1, attempts)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Codex masked object insertion failed", exc_info=True)
        return _error(f"Masked object insertion via Codex auth failed: {exc}", "api_error", prompt)

    if not collected or not collected.get("b64"):
        return _error(f"Codex response contained no image_generation_call result after {attempts} attempt(s)",
                      "empty_response", prompt)
    if collected.get("source") != "final":
        return _error(f"Codex returned only a progressive partial image frame after {attempts} attempt(s); "
                      "refusing to save it as a final deliverable.", "incomplete_image", prompt)

    b64 = collected["b64"]
    try:
        from agent.image_gen_provider import save_b64_image
        pixel_size = _png_pixel_size(base64.b64decode(b64))
        saved_path = save_b64_image(b64, prefix="codex_image_mask_ref")
    except Exception as exc:  # noqa: BLE001
        return _error(f"Could not save image to cache: {exc}", "io_error", prompt)
    return {
        "success": True, "image": str(saved_path), "model": IMAGE_TIER, "prompt": prompt,
        "modality": "image", "provider": PROVIDER, "action": IMAGE_ACTION,
        "quality": IMAGE_QUALITY, "input_fidelity": input_fidelity, "mask_mode": mode,
        "image_roles": _IMAGE_ROLES[mode], "input_size": validation.get("size"),
        "pixel_size": pixel_size,
    }


def _handle_codex_image_mask_with_reference(args: Dict[str, Any], **kw: Any) -> str:
    prompt = args.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return tool_error("prompt is required for masked object insertion")
    sources = {}
    for key, label in (("image_url", "the main image to edit"), ("mask_url", "the mask image"),
                       ("object_url", "the object reference image")):
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            return tool_error(f"{key} ({label}) is required")
        sources[key] = value
    task_id = kw.get("task_id")
    # Same sandbox confinement as image_generate: under a non-local terminal backend, path-like
    # sources are resolved to data: URLs through the vision chokepoint.
    from tools.image_generation_tool import _confine_source_images, _postprocess_image_generate_result
    image_url, refs, confine_error = _confine_source_images(
        sources["image_url"], [sources["mask_url"], sources["object_url"]], task_id)
    if confine_error is not None:
        return confine_error
    mask_url, object_url = (tuple(refs) if isinstance(refs, (list, tuple)) and len(refs) == 2
                            else (sources["mask_url"], sources["object_url"]))
    raw = json.dumps(codex_image_mask_with_reference_tool(prompt, image_url, mask_url, object_url),
                     indent=2, ensure_ascii=False)
    return _postprocess_image_generate_result(raw, task_id=task_id)


CODEX_IMAGE_MASK_WITH_REFERENCE_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Insert an object from a reference image into a masked region of a main image "
        "(mask-guided compositing) using OpenAI gpt-image-2 at high quality via ChatGPT/Codex "
        "auth. Always takes a prompt plus THREE images: the main scene, a mask of the same "
        "pixel size whose TRANSPARENT pixels mark where the object goes (opaque pixels are "
        "preserved), and an object reference image whose subject is extracted and placed "
        "there. The tool automatically tells the model which attached image is which, so "
        "describe images by role ('the main scene', 'the mask', 'the object image') rather "
        "than by number. Returns the edited PNG's absolute file path in the `image` field; "
        "reference it in your response using the current platform's file-delivery convention."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "How to place the object inside the masked region and blend it (lighting, "
                    "contact shadows, scale, orientation) while keeping its exact details. "
                    "Be detailed and descriptive."
                ),
            },
            "image_url": {
                "type": "string",
                "description": (
                    "The MAIN scene image to edit: an absolute local file path, a public URL, "
                    "or a base64 data:image URL. PNG, JPEG, WebP or GIF, max 25MB."
                ),
            },
            "mask_url": {
                "type": "string",
                "description": (
                    "The MASK image: a PNG with an alpha channel, same width and height as the "
                    "main image. Transparent pixels = placement zone; opaque = keep unchanged. "
                    "Absolute local path, public URL, or data:image URL."
                ),
            },
            "object_url": {
                "type": "string",
                "description": (
                    "The OBJECT reference image containing the physical object to extract and "
                    "place in the masked region (any size). Absolute local path, public URL, "
                    "or data:image URL."
                ),
            },
        },
        "required": ["prompt", "image_url", "mask_url", "object_url"],
    },
}


registry.register(
    name=TOOL_NAME, toolset="image_gen", schema=CODEX_IMAGE_MASK_WITH_REFERENCE_SCHEMA,
    handler=_handle_codex_image_mask_with_reference,
    check_fn=check_codex_image_mask_with_reference_requirements,
    requires_env=[], is_async=False, emoji="🧩",
)
