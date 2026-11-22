#!/usr/bin/env python3
"""Mask-guided object insertion via the ChatGPT/Codex Responses API with SEVERAL object references.

Same contract as ``tools/codex_image_mask_with_reference.py`` (``gpt-image-2``, ``action: "edit"``,
``input_fidelity`` with the same backend fallback, mask as ``input_image_mask`` in ``tool`` mode or
as an extra image in ``inline`` mode) but takes a LIST of object reference images, each with an
optional label (e.g. "the barn from the front", "the barn from directly above"). Typical use: several
photos of the same object from different sides, so the model does not have to guess the parts a
single photo cannot show.

Content parts, in order: the role preamble + prompt, the main scene (the mask applies to the FIRST
image, per the OpenAI image generation guide), [the mask in ``inline`` mode], then every object
reference. The preamble names each attached image by number and label, and is echoed back in the
result as ``image_roles``. All shared plumbing is imported from ``tools.codex_image_mask``.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from tools.codex_image_mask import (
    IMAGE_ACTION, IMAGE_MODEL, IMAGE_QUALITY, IMAGE_TIER, INPUT_FIDELITY, PROVIDER,
    _CODEX_CHAT_MODEL, _NO_AUTH, _NONFINAL_RETRIES, _PARTIAL_IMAGES_REQUESTED,
    _collect_image_b64, _httpx_available, _load_image_bytes, _mask_mode, _png_pixel_size,
    _read_codex_access_token, _rejects_input_fidelity, _to_data_url, _validate_pair,
)
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

TOOL_NAME = "codex_image_mask_with_references"
MAX_OBJECTS = 15  # 16 input images in total with the main scene

_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image editing requests by using the "
    "image_generation tool when provided. The user supplies a main scene image, a mask that "
    "defines the only editable region, and one or more reference images of an object (often the "
    "same object from different sides). Use all reference images together to understand the "
    "object exactly, then place it inside the masked region of the main scene as instructed, "
    "preserving the object's exact details and keeping everything outside the mask unchanged.")


def check_codex_image_mask_with_references_requirements() -> bool:
    return bool(_read_codex_access_token()) and _httpx_available()


def _image_roles(mode: str, labels: Sequence[str]) -> str:
    """Deterministic preamble naming every attached image in the order the host model sees them."""
    parts = ["(1) the main scene to edit"]
    n = 2
    if mode == "inline":
        parts.append("(2) the mask: its transparent pixels mark the ONLY region you may repaint, "
                     "opaque pixels must stay unchanged")
        n = 3
    for i, label in enumerate(labels):
        parts.append(f"({n + i}) object reference {i + 1}: {label}")
    text = "Attached images, in order: " + "; ".join(parts) + "."
    if mode != "inline":
        text += (" The editable region is defined by the provided image mask (transparent pixels); "
                 "everything outside it must stay unchanged.")
    return text


def _build_responses_payload(
    *, prompt: str, main_data_url: str, mask_data_url: str, object_data_urls: Sequence[str],
    roles: str, mode: str,
) -> Dict[str, Any]:
    """Responses body for one masked object-insertion edit with several references. No
    ``tool_choice``: the Codex backend rejects every forcing shape."""
    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": f"{roles}\n\n{prompt}"},
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
        "background": "opaque",
        "partial_images": _PARTIAL_IMAGES_REQUESTED,
    }
    if mode == "inline":
        content.append({"type": "input_image", "image_url": mask_data_url})
    else:
        tool["input_image_mask"] = {"image_url": mask_data_url}
    for url in object_data_urls:
        content.append({"type": "input_image", "image_url": url})
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


def codex_image_mask_with_references_tool(
    prompt: str, image_url: str, mask_url: str, object_urls: Sequence[str],
    object_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Place the object shown in ``object_urls`` (one or more pictures of it, optionally labelled
    with ``object_labels``) inside the ``mask_url`` region of ``image_url`` according to
    ``prompt``. Returns the same ``{"success", "image", ...}`` dict shape as
    ``codex_image_mask_with_reference``."""
    prompt = (prompt or "").strip()
    if not prompt:
        return _error("Prompt is required and must be a non-empty string", "invalid_argument")
    object_urls = [u for u in (object_urls or []) if isinstance(u, str) and u.strip()]
    if not object_urls:
        return _error("At least one object reference image is required", "invalid_argument", prompt)
    if len(object_urls) > MAX_OBJECTS:
        return _error(f"At most {MAX_OBJECTS} object reference images are supported", "invalid_argument", prompt)
    labels = list(object_labels or [])
    if len(labels) > len(object_urls):
        return _error("More object labels than object images", "invalid_argument", prompt)
    labels = [str(l).strip() or "the object" for l in labels]
    labels += ["the object"] * (len(object_urls) - len(labels))
    token = _read_codex_access_token()
    if not token:
        return _error(_NO_AUTH, "auth_required", prompt)
    if not _httpx_available():
        return _error("httpx Python package not installed (pip install httpx)", "missing_dependency", prompt)

    try:
        main_raw = _load_image_bytes(image_url, "image_url (main image)")
        mask_raw = _load_image_bytes(mask_url, "mask_url (mask image)")
        validation = _validate_pair(main_raw, mask_raw)
        main_data_url = _to_data_url(main_raw, "image_url (main image)")
        mask_data_url = _to_data_url(mask_raw, "mask_url (mask image)")
        object_data_urls = []
        for i, url in enumerate(object_urls, 1):
            label = f"object_urls[{i}] (object reference {i})"
            object_data_urls.append(_to_data_url(_load_image_bytes(url, label), label))
    except Exception as exc:  # noqa: BLE001
        return _error(f"Invalid image input for masked object insertion: {exc}", "invalid_image_input", prompt)

    mode = _mask_mode()
    roles = _image_roles(mode, labels)
    payload = _build_responses_payload(
        prompt=prompt, main_data_url=main_data_url, mask_data_url=mask_data_url,
        object_data_urls=object_data_urls, roles=roles, mode=mode)
    attempts = _NONFINAL_RETRIES + 1
    input_fidelity: str = INPUT_FIDELITY
    logger.info("Masked object insertion via Codex (%s, quality=%s, mask_mode=%s, %d references) — prompt: %s",
                IMAGE_MODEL, IMAGE_QUALITY, mode, len(object_urls), prompt[:80])
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
        logger.debug("Codex masked object insertion (references) failed", exc_info=True)
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
        saved_path = save_b64_image(b64, prefix="codex_image_mask_refs")
    except Exception as exc:  # noqa: BLE001
        return _error(f"Could not save image to cache: {exc}", "io_error", prompt)
    return {
        "success": True, "image": str(saved_path), "model": IMAGE_TIER, "prompt": prompt,
        "modality": "image", "provider": PROVIDER, "action": IMAGE_ACTION,
        "quality": IMAGE_QUALITY, "input_fidelity": input_fidelity, "mask_mode": mode,
        "image_roles": roles, "object_count": len(object_urls), "input_size": validation.get("size"),
        "pixel_size": pixel_size,
    }


def _handle_codex_image_mask_with_references(args: Dict[str, Any], **kw: Any) -> str:
    prompt = args.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return tool_error("prompt is required for masked object insertion")
    image_url, mask_url = args.get("image_url"), args.get("mask_url")
    if not isinstance(image_url, str) or not image_url.strip():
        return tool_error("image_url (the main image to edit) is required")
    if not isinstance(mask_url, str) or not mask_url.strip():
        return tool_error("mask_url (the mask image) is required")
    object_urls = args.get("object_urls")
    if not isinstance(object_urls, list) or not object_urls or not all(isinstance(u, str) and u.strip() for u in object_urls):
        return tool_error("object_urls (a list of one or more object reference images) is required")
    object_labels = args.get("object_labels") or []
    if not isinstance(object_labels, list):
        return tool_error("object_labels must be a list of strings")
    task_id = kw.get("task_id")
    # Same sandbox confinement as image_generate: under a non-local terminal backend, path-like
    # sources are resolved to data: URLs through the vision chokepoint.
    from tools.image_generation_tool import _confine_source_images, _postprocess_image_generate_result
    image_url, refs, confine_error = _confine_source_images(image_url, [mask_url, *object_urls], task_id)
    if confine_error is not None:
        return confine_error
    if isinstance(refs, (list, tuple)) and len(refs) == 1 + len(object_urls):
        mask_url, object_urls = refs[0], list(refs[1:])
    raw = json.dumps(codex_image_mask_with_references_tool(prompt, image_url, mask_url, object_urls, object_labels),
                     indent=2, ensure_ascii=False)
    return _postprocess_image_generate_result(raw, task_id=task_id)


CODEX_IMAGE_MASK_WITH_REFERENCES_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Insert an object into a masked region of a main image using OpenAI gpt-image-2 at high "
        "quality via ChatGPT/Codex auth, using SEVERAL reference images of the object (e.g. the same "
        "building from the front, back, side and above). Takes a prompt, the main scene, a mask of "
        "the same pixel size whose TRANSPARENT pixels mark where the object goes (opaque pixels are "
        "preserved), and 1-15 object reference images with optional labels. The tool tells the model "
        "which attached image is which. Returns the edited PNG's absolute file path in the `image` "
        "field; reference it in your response using the current platform's file-delivery convention."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "How to place the object inside the masked region and blend it in.",
            },
            "image_url": {
                "type": "string",
                "description": ("The MAIN scene image to edit: absolute local path, public URL, or base64 "
                                "data:image URL. PNG, JPEG, WebP or GIF, max 25MB."),
            },
            "mask_url": {
                "type": "string",
                "description": ("The MASK: PNG with an alpha channel, same size as the main image. "
                                "Transparent = placement zone; opaque = keep unchanged."),
            },
            "object_urls": {
                "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": MAX_OBJECTS,
                "description": "One or more reference images of the object (any size).",
            },
            "object_labels": {
                "type": "array", "items": {"type": "string"},
                "description": ("Optional label per object image, in the same order, e.g. 'the barn from "
                                "the front', 'the barn from directly above'."),
            },
        },
        "required": ["prompt", "image_url", "mask_url", "object_urls"],
    },
}


registry.register(
    name=TOOL_NAME, toolset="image_gen", schema=CODEX_IMAGE_MASK_WITH_REFERENCES_SCHEMA,
    handler=_handle_codex_image_mask_with_references,
    check_fn=check_codex_image_mask_with_references_requirements,
    requires_env=[], is_async=False, emoji="🧩",
)
