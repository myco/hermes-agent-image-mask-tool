#!/usr/bin/env python3
"""Mask-guided image editing (inpainting) via the ChatGPT/Codex Responses API.

Fixed contract, no knobs: ``prompt`` + main image + mask image → one edited PNG. Every call
uses the ``image_generation`` hosted tool with ``model: gpt-image-2`` at ``quality: high`` (the
``gpt-image-2-high`` picker tier), ``action: "edit"`` and ``input_fidelity: "high"``. Auth reuses
the Codex OAuth token (``hermes auth codex``) exactly like ``plugins/image_gen/openai-codex``;
no ``OPENAI_API_KEY`` is needed.

Known backend quirk: chatgpt.com/backend-api/codex serves ``gpt-image-2`` as an internal
``gpt-image-2-codex`` variant that 400s on ``input_fidelity``. The knob is always sent first;
on that specific rejection the call is retried once without it and the result reports
``"input_fidelity": "unsupported_by_backend"`` so the degradation is visible, never silent.

Mask wiring follows the documented Responses shape: the main image travels as an
``input_image`` content part and the mask as the tool's ``input_image_mask``. Set
``CODEX_IMAGE_MASK_MODE=inline`` to send the mask as a *second* ``input_image`` part instead
(some backends only honour that form). Transparent mask pixels mark the region to repaint and
the mask must match the main image's pixel size — both are validated up front so the API never
sees a request it will reject anyway.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

TOOL_NAME = "codex_image_mask"
PROVIDER = "openai-codex"

# Fixed generation contract (see module docstring).
IMAGE_MODEL = "gpt-image-2"
IMAGE_TIER = "gpt-image-2-high"
IMAGE_QUALITY = "high"
IMAGE_ACTION = "edit"
INPUT_FIDELITY = "high"

# Hosts the ``image_generation`` tool call; ``IMAGE_MODEL`` does the image work.
_CODEX_CHAT_MODEL = "gpt-5.5"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image editing requests by using the "
    "image_generation tool when provided. The user supplies a source image and a mask; "
    "edit only the masked region as instructed and keep everything else unchanged.")

_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
_ACCEPTED_INPUT_MIME = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_PARTIAL_IMAGES_REQUESTED = 0
_NONFINAL_RETRIES = 1  # content-agnostic retries when the stream yields no final result
_MASK_MODES = ("tool", "inline")

_NO_AUTH = (
    "No Codex/ChatGPT OAuth credentials available. Run "
    "`hermes auth codex` (or `hermes setup` → Codex) to sign in.")


# --- Auth / dependencies ---
def _read_codex_access_token() -> Optional[str]:
    """Usable Codex OAuth token or None (``agent.auxiliary_client`` owns expiry/pool/JWT)."""
    try:
        from agent.auxiliary_client import _read_codex_access_token as _reader
        token = _reader()
        return token.strip() if isinstance(token, str) and token.strip() else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not resolve Codex access token: %s", exc)
        return None


def _httpx_available() -> bool:
    try:
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


def check_codex_image_mask_requirements() -> bool:
    return bool(_read_codex_access_token()) and _httpx_available()


def _mask_mode() -> str:
    mode = (os.environ.get("CODEX_IMAGE_MASK_MODE") or "tool").strip().lower()
    return mode if mode in _MASK_MODES else "tool"


# --- Input images → bytes → data URLs ---
def _sniff_image_mime(raw: bytes) -> Optional[str]:
    from agent.image_routing import _sniff_mime_from_bytes
    mime = _sniff_mime_from_bytes(raw)
    return mime if mime in _ACCEPTED_INPUT_MIME else None


def _load_image_bytes(value: str, label: str) -> bytes:
    """Raw bytes for a local path / ``data:`` URL / http(s) URL, size-capped."""
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError(f"{label} is required")
    lowered = candidate.lower()
    if lowered.startswith("data:"):
        if "," not in candidate:
            raise ValueError(f"{label} data URL is missing a comma separator")
        header, data = candidate.split(",", 1)
        if not header.lower().startswith("data:image/") or ";base64" not in header.lower():
            raise ValueError(f"{label} must be a base64 data:image URL")
        raw = base64.b64decode(data, validate=True)
    elif lowered.startswith(("http://", "https://")):
        import httpx
        with httpx.Client(timeout=60.0, follow_redirects=True) as http:
            response = http.get(candidate)
            response.raise_for_status()
            raw = response.content
    else:
        try:
            from agent.file_safety import get_read_block_error
            blocked = get_read_block_error(candidate)
            if blocked:
                raise ValueError(blocked)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - guard is best-effort
            logger.debug("Image input read guard unavailable: %s", exc)
        path = Path(os.path.expanduser(candidate)).resolve()
        if not path.is_file():
            raise ValueError(f"{label} path does not exist or is not a file: {value}")
        raw = path.read_bytes()
    if not raw:
        raise ValueError(f"{label} is empty: {value}")
    if len(raw) > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"{label} exceeds 25MB cap: {value}")
    return raw


def _to_data_url(raw: bytes, label: str) -> str:
    mime = _sniff_image_mime(raw)
    if mime is None:
        raise ValueError(f"{label} is not a supported image (png/jpeg/gif/webp)")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _validate_pair(main_raw: bytes, mask_raw: bytes) -> Dict[str, Any]:
    """Enforce the API's mask rules (same pixel size, alpha channel) before spending a call.
    Returns ``{"size", "mask_has_alpha"}``; skipped (empty dict) when Pillow is missing."""
    try:
        from PIL import Image
    except ImportError:
        logger.debug("Pillow not installed; skipping mask/image validation")
        return {}
    with Image.open(io.BytesIO(main_raw)) as main_img, Image.open(io.BytesIO(mask_raw)) as mask_img:
        if main_img.size != mask_img.size:
            raise ValueError(
                f"Mask must have the same pixel size as the main image: main is "
                f"{main_img.size[0]}x{main_img.size[1]}, mask is {mask_img.size[0]}x{mask_img.size[1]}")
        has_alpha = mask_img.mode in ("RGBA", "LA", "PA") or "transparency" in mask_img.info
        if not has_alpha:
            raise ValueError(
                "Mask image has no alpha channel. The mask must be a PNG whose transparent pixels "
                "mark the region to edit (opaque pixels are preserved).")
        return {"size": f"{main_img.size[0]}x{main_img.size[1]}", "mask_has_alpha": True}


# --- Responses payload + streaming ---
def _build_responses_payload(*, prompt: str, main_data_url: str, mask_data_url: str, mode: str) -> Dict[str, Any]:
    """Responses body for one masked ``image_generation`` edit. No ``tool_choice``: the Codex
    backend rejects every shape for forcing the hosted tool, so ``instructions`` nudge the host."""
    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
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
    return {
        "model": _CODEX_CHAT_MODEL,
        "store": False,
        "instructions": _CODEX_INSTRUCTIONS,
        "input": [{"type": "message", "role": "user", "content": content}],
        "tools": [tool],
        "stream": True,
    }


def _summarize_error_body(body: str, limit: int = 500) -> str:
    """Bounded summary preferring parsed ``error.message`` (Codex bodies carry leading metadata)."""
    text = body or ""
    try:
        payload = json.loads(text)
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str) and message.strip():
            return message.strip()[:limit]
    except (TypeError, ValueError):
        pass
    return text[:limit]


def _iter_sse_json(response: Any):
    """JSON payloads from an SSE response, without SDK parsing."""
    event_name: Optional[str] = None
    data_lines: List[str] = []

    def flush():
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        raw = "\n".join(data_lines).strip()
        event, event_name, data_lines = event_name, None, []
        if not raw or raw == "[DONE]":
            return None
        payload = json.loads(raw)
        if isinstance(payload, dict) and event and "type" not in payload:
            payload["type"] = event
        return payload

    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = str(line)
        if line == "":
            payload = flush()
            if payload is not None:
                yield payload
        elif line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    payload = flush()
    if payload is not None:
        yield payload


def _extract_image_candidates(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """``(final_result_b64, latest_partial_b64)`` from a payload tree."""
    result_b64: Optional[str] = None
    partial_b64: Optional[str] = None

    def walk(node: Any) -> None:
        nonlocal result_b64, partial_b64
        if isinstance(node, dict):
            result = node.get("result") if node.get("type") == "image_generation_call" else None
            if isinstance(result, str) and result:
                result_b64 = result
            partial = node.get("partial_image_b64")
            if isinstance(partial, str) and partial:
                partial_b64 = partial
        for child in node.values() if isinstance(node, dict) else node if isinstance(node, list) else ():
            walk(child)

    walk(value)
    return result_b64, partial_b64


def _collect_image_b64(token: str, payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Stream one Codex Responses call → ``{"b64", "source": "final"|"partial"}`` or ``None``."""
    import httpx
    from agent.codex_headers import codex_cloudflare_headers

    headers = codex_cloudflare_headers(token)
    headers.update({
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=30.0, pool=30.0)
    final_b64: Optional[str] = None
    partial_b64: Optional[str] = None
    with httpx.Client(timeout=timeout, headers=headers) as http:
        with http.stream("POST", f"{_CODEX_BASE_URL}/responses", json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                exc.response.read()
                raise RuntimeError(
                    f"Codex Responses API returned HTTP {exc.response.status_code}: "
                    f"{_summarize_error_body(exc.response.text)}") from exc
            for event in _iter_sse_json(response):
                result_b64, event_partial = _extract_image_candidates(event)
                final_b64 = result_b64 or final_b64
                partial_b64 = event_partial or partial_b64
    if final_b64:
        return {"b64": final_b64, "source": "final"}
    return {"b64": partial_b64, "source": "partial"} if partial_b64 else None


def _rejects_input_fidelity(exc: BaseException) -> bool:
    """True for the Codex 400 ``The model '...' does not support the 'input_fidelity' parameter``."""
    text = str(exc).lower()
    return "input_fidelity" in text and "http 400" in text


def _png_pixel_size(raw: bytes) -> Optional[str]:
    import struct
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return f"{width}x{height}"


# --- Tool entry point ---
def _error(error: str, error_type: str, prompt: str = "", **extra: Any) -> Dict[str, Any]:
    return {"success": False, "image": None, "error": error, "error_type": error_type,
            "model": IMAGE_TIER, "prompt": prompt, "provider": PROVIDER, **extra}


def codex_image_mask_tool(prompt: str, image_url: str, mask_url: str) -> Dict[str, Any]:
    """Edit ``image_url`` inside the region marked by ``mask_url`` according to ``prompt``.
    Returns the same ``{"success", "image", ...}`` dict shape as ``image_generate``."""
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
        validation = _validate_pair(main_raw, mask_raw)
        main_data_url = _to_data_url(main_raw, "image_url (main image)")
        mask_data_url = _to_data_url(mask_raw, "mask_url (mask image)")
    except Exception as exc:  # noqa: BLE001
        return _error(f"Invalid image input for masked edit: {exc}", "invalid_image_input", prompt)

    mode = _mask_mode()
    payload = _build_responses_payload(
        prompt=prompt, main_data_url=main_data_url, mask_data_url=mask_data_url, mode=mode)
    attempts = _NONFINAL_RETRIES + 1
    input_fidelity: str = INPUT_FIDELITY
    logger.info("Masked edit via Codex (%s, quality=%s, mask_mode=%s) — prompt: %s",
                IMAGE_MODEL, IMAGE_QUALITY, mode, prompt[:80])
    try:
        collected: Optional[Dict[str, str]] = None
        for attempt in range(attempts):
            try:
                collected = _collect_image_b64(token, payload)
            except RuntimeError as exc:
                # The Codex backend serves gpt-image-2 as an internal variant that rejects
                # ``input_fidelity`` with a 400; drop just that knob and retry rather than fail.
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
        logger.debug("Codex masked edit failed", exc_info=True)
        return _error(f"Masked image edit via Codex auth failed: {exc}", "api_error", prompt)

    if not collected or not collected.get("b64"):
        return _error(f"Codex response contained no image_generation_call result after {attempts} attempt(s)",
                      "empty_response", prompt)
    if collected.get("source") != "final":
        # Never deliver a progressive-only frame as success (smeared previews).
        return _error(f"Codex returned only a progressive partial image frame after {attempts} attempt(s); "
                      "refusing to save it as a final deliverable.", "incomplete_image", prompt)

    b64 = collected["b64"]
    try:
        from agent.image_gen_provider import save_b64_image
        pixel_size = _png_pixel_size(base64.b64decode(b64))
        saved_path = save_b64_image(b64, prefix="codex_image_mask")
    except Exception as exc:  # noqa: BLE001
        return _error(f"Could not save image to cache: {exc}", "io_error", prompt)
    return {
        "success": True, "image": str(saved_path), "model": IMAGE_TIER, "prompt": prompt,
        "modality": "image", "provider": PROVIDER, "action": IMAGE_ACTION,
        "quality": IMAGE_QUALITY, "input_fidelity": input_fidelity, "mask_mode": mode,
        "input_size": validation.get("size"), "pixel_size": pixel_size,
    }


def _handle_codex_image_mask(args: Dict[str, Any], **kw: Any) -> str:
    prompt = args.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return tool_error("prompt is required for masked image editing")
    image_url, mask_url = args.get("image_url"), args.get("mask_url")
    if not isinstance(image_url, str) or not image_url.strip():
        return tool_error("image_url (the main image to edit) is required")
    if not isinstance(mask_url, str) or not mask_url.strip():
        return tool_error("mask_url (the mask image) is required")
    task_id = kw.get("task_id")
    # Same sandbox confinement as image_generate: under a non-local terminal backend, path-like
    # sources are resolved to data: URLs through the vision chokepoint.
    from tools.image_generation_tool import _confine_source_images, _postprocess_image_generate_result
    image_url, refs, confine_error = _confine_source_images(image_url, [mask_url], task_id)
    if confine_error is not None:
        return confine_error
    mask_url = refs[0] if isinstance(refs, (list, tuple)) and refs else mask_url
    raw = json.dumps(codex_image_mask_tool(prompt, image_url, mask_url), indent=2, ensure_ascii=False)
    return _postprocess_image_generate_result(raw, task_id=task_id)


CODEX_IMAGE_MASK_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Edit a region of an existing image guided by a mask (inpainting) using OpenAI "
        "gpt-image-2 at high quality via ChatGPT/Codex auth. Always takes a prompt plus TWO "
        "images: the main image to edit and a mask of the same pixel size whose TRANSPARENT "
        "pixels mark the area to repaint (opaque pixels are preserved with high input "
        "fidelity). Returns the edited PNG's absolute file path in the `image` field; "
        "reference it in your response using the current platform's file-delivery convention."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "What to paint inside the masked region (and how it should blend with "
                    "the rest of the image). Be detailed and descriptive."
                ),
            },
            "image_url": {
                "type": "string",
                "description": (
                    "The MAIN image to edit: an absolute local file path, a public URL, or a "
                    "base64 data:image URL. PNG, JPEG, WebP or GIF, max 25MB."
                ),
            },
            "mask_url": {
                "type": "string",
                "description": (
                    "The MASK image: a PNG with an alpha channel, same width and height as the "
                    "main image. Transparent pixels = region to edit; opaque = keep unchanged. "
                    "Absolute local path, public URL, or data:image URL."
                ),
            },
        },
        "required": ["prompt", "image_url", "mask_url"],
    },
}


registry.register(
    name=TOOL_NAME, toolset="image_gen", schema=CODEX_IMAGE_MASK_SCHEMA,
    handler=_handle_codex_image_mask, check_fn=check_codex_image_mask_requirements,
    requires_env=[], is_async=False, emoji="🎭",
)
