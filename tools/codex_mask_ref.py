#!/home/rafal/.hermes/hermes-agent/venv/bin/python
"""Run Hermes' Codex masked image-edit tools outside the agent.

With -r/--object: codex_image_mask_with_reference (main + mask + object reference image).
Without it:       codex_image_mask                (main + mask only).

Both go through a Codex chat model (--chat-model) that calls gpt-image-2's image_generation
tool; the mask is passed to that tool, which uses it to decide where the edit goes.

Imports the tools from the Hermes source tree and calls them as-is, so they use the same
ChatGPT/Codex OAuth token Hermes uses (resolved by agent.auxiliary_client from ~/.hermes).

To keep the new object inside the mask, the prompt gets a size limit derived from the mask,
and each result is checked for spill-over just outside the mask; if it spilled, the edit is
retried asking for a smaller object (see --attempts, --max-spill, --no-size-hint).

Usage:
    ./codex_mask_ref.py                                   # mask-only test run with image-edit-sunset PNGs
    ./codex_mask_ref.py -r ~/image-edit-sunset/object.png # reference test run
    ./codex_mask_ref.py -p "prompt" -i main.png -m mask.png [-r object.png] -o out.png

    From Windows (C:\\Users\\Rafal\\claude_boards is mounted at /mnt/claude_boards):
    wsl.exe -d Ubuntu -- /home/rafal/codex_mask_ref.py --composite -p "..." \\
        -i /mnt/claude_boards/main.png -m /mnt/claude_boards/mask.png -o /mnt/claude_boards/result.png

How to use it well:
  * Mask: a PNG with the same pixel size as the main image. TRANSPARENT pixels = the area to
    edit, opaque pixels = keep. (Image viewers often show transparent pixels as white.)
  * Prompt: always say that the object goes "inside the masked area". The chat model that
    writes the image instructions does not see the mask (default --mode tool), so also describe
    the area by what is visible there, e.g. "on the pool water", "on the bright green ground",
    and say what to keep. Describe size relative to the scene ("shorter than the coffee table")
    rather than in percentages, and avoid words like "big".
  * Objects (a boat, a cactus, a phone): use the defaults. A size limit computed from the mask is
    appended to the prompt, each result is checked for spill-over just outside the mask, and up
    to --attempts tries are made, asking for a smaller object each time.
  * Non-objects that fill the whole masked area (landscapes, textures, terrain, sky, floors):
    add --no-size-hint --attempts 1. The size limit makes no sense there, and the spill check
    gives false alarms because the model redraws the mask border.
  * --composite: after generation, keeps the generated pixels only inside the mask and restores
    everything outside it from the original image (the model otherwise redraws the whole picture
    slightly). The output has the generated image's resolution; if that differs from the
    original's, the original and mask are scaled to it first, so restored areas are the original
    rescaled rather than pixel-identical. Anything the model drew past the mask edge is cut off, so make
    the mask big enough for the whole object. Recommended for almost every run.
  * Mask with several separate areas: the model tends to fill only one of them. Ask for each
    area explicitly in the prompt (with --no-size-hint), or run once per area.
  * Exit codes: 0 = OK, 1 = generation failed, 2 = saved, but every attempt spilled past the
    mask (or the spill check misfired on a non-object edit).
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageChops, ImageFilter

HERMES_DIR = os.path.expanduser("~/.hermes/hermes-agent")
TEST_DIR = os.path.expanduser("~/image-edit-sunset")

DEFAULT_PROMPT_REFERENCE = (
    "Place the red rotary telephone from the object image inside the masked box of the main "
    "scene. Scale it to fit neatly within the box, keep its exact shape, colors and dial "
    "details, and add a soft contact shadow matching the flat illustration's lighting. "
    "Leave every other box and the background unchanged."
)
DEFAULT_PROMPT_MASK = (
    "Draw a small green potted plant in a terracotta pot inside the masked box, centered and "
    "scaled to fit neatly within the box, with a soft contact shadow matching the flat "
    "illustration's lighting. Leave every other box and the background unchanged."
)

SPILL_BAND_PX = 20   # width of the ring just outside the mask that is checked for spill-over
SPILL_DIFF = 40      # per-pixel change (0-255, after a light blur) that counts as "changed"


def editable_area(mask_path: str) -> Image.Image:
    """L-mode image: 255 where the mask is transparent (editable), 0 elsewhere."""
    alpha = Image.open(mask_path).convert("RGBA").getchannel("A")
    return alpha.point(lambda v: 255 if v < 128 else 0)


def size_hint(mask_path: str) -> str:
    """Prompt text stating how big the masked area is, as fractions of the image."""
    editable = editable_area(mask_path)
    x0, y0, x1, y1 = editable.getbbox()
    w, h = editable.size
    pct = lambda v, total: round(100 * v / total)
    return (
        f"SIZE LIMIT: the masked (editable) area is small: about {pct(x1 - x0, w)}% of the image "
        f"width and {pct(y1 - y0, h)}% of the image height, spanning {pct(x0, w)}-{pct(x1, w)}% "
        f"from the left edge and {pct(y0, h)}-{pct(y1, h)}% from the top edge. The new object "
        "must fit ENTIRELY inside that area with some empty space around it; no part of it may "
        "extend outside. Make the object smaller than its natural size if needed to fit."
    )


def spill_percent(original_path: str, mask_path: str, generated_path: str) -> float:
    """% of pixels changed in a thin ring just outside the mask, i.e. how much of the new object
    spilled past the mask edge. Background noise from full-image regeneration is ~0-0.5%."""
    original = Image.open(original_path).convert("RGB")
    generated = Image.open(generated_path).convert("RGB").resize(original.size, Image.LANCZOS)
    editable = editable_area(mask_path)
    grown = editable
    for _ in range(SPILL_BAND_PX // 2):
        grown = grown.filter(ImageFilter.MaxFilter(5))
    ring = (np.array(grown) > 0) & ~(np.array(editable) > 0)
    diff = ImageChops.difference(original.filter(ImageFilter.GaussianBlur(2)),
                                 generated.filter(ImageFilter.GaussianBlur(2))).convert("L")
    changed = np.array(diff) > SPILL_DIFF
    return round(100 * float(changed[ring].mean()), 2)


def composite(original_path: str, mask_path: str, generated_path: str, out_path: str, feather: float) -> None:
    """Keep generated pixels only inside the mask's transparent area, original pixels elsewhere.
    The output always has the generated image's resolution: the original and the mask are scaled
    (up or down) to it, since the backend may return a different size than the original."""
    original = Image.open(original_path).convert("RGB")
    generated = Image.open(generated_path).convert("RGB")
    scale = generated.width / original.width
    rescaled = generated.size != original.size
    if not rescaled:
        editable = editable_area(mask_path)
    else:
        original = original.resize(generated.size, Image.LANCZOS)
        # Resize the soft alpha before thresholding so the mask edge stays smooth.
        alpha = Image.open(mask_path).convert("RGBA").getchannel("A").resize(generated.size, Image.BILINEAR)
        editable = alpha.point(lambda v: 255 if v < 128 else 0)
    # Grow the editable area so the soft edge (and resampling blur of a rescaled original) falls on
    # the kept side: original pixels from the editable area must never show through at the border.
    grow = math.ceil(2 * feather) + (math.ceil(max(scale, 1)) if rescaled else 0)
    if grow > 0:
        editable = editable.filter(ImageFilter.MaxFilter(2 * grow + 1))
    if feather > 0:
        editable = editable.filter(ImageFilter.GaussianBlur(feather))
    Image.composite(generated, original, editable).save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--prompt", help="edit prompt (default: a test prompt for the chosen tool)")
    ap.add_argument("-i", "--image", default=f"{TEST_DIR}/input.png", help="main scene image")
    ap.add_argument("-m", "--mask", default=f"{TEST_DIR}/mask.png", help="mask PNG (transparent = edit zone)")
    ap.add_argument("-r", "--object",
                    help="object reference image; if omitted, codex_image_mask (main + mask only) is used")
    ap.add_argument("-o", "--output", default=f"{TEST_DIR}/output.png",
                    help="where to copy the result (Hermes also keeps a copy in ~/.hermes/cache/images)")
    ap.add_argument("--mode", choices=["tool", "inline"],
                    help="mask mode (sets CODEX_IMAGE_MASK_MODE; default: tool)")
    ap.add_argument("--chat-model", default="gpt-5.6-sol",
                    help="Codex chat model that calls image_generation (Hermes hardcodes gpt-5.5, "
                         "which this ChatGPT account rejects); pass 'hermes' to keep Hermes' value")
    ap.add_argument("--attempts", type=int, default=3,
                    help="max generations; each retry asks for a ~30%% smaller object (default: 3, "
                         "1 = no retries)")
    ap.add_argument("--max-spill", type=float, default=1.5,
                    help="max %% of changed pixels allowed in a 20px ring outside the mask before "
                         "the result counts as spilled (default: 1.5)")
    ap.add_argument("--no-size-hint", action="store_true",
                    help="don't append the mask-derived size limit to the prompt")
    ap.add_argument("--composite", action="store_true",
                    help="paste the result back into the original inside the mask only, so nothing "
                         "outside the mask changes (objects overflowing the mask get clipped); output "
                         "always has the generated image's resolution")
    ap.add_argument("--feather", type=float, default=3.0,
                    help="blur radius in px for the --composite mask edge (default: 3)")
    args = ap.parse_args()

    if args.mode:
        os.environ["CODEX_IMAGE_MASK_MODE"] = args.mode
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Make Hermes packages (tools.*, agent.*) importable.
    sys.path.insert(0, HERMES_DIR)
    image, mask = os.path.abspath(args.image), os.path.abspath(args.mask)

    if args.object:
        import tools.codex_image_mask_with_reference as tool_mod
        call = lambda prompt: tool_mod.codex_image_mask_with_reference_tool(
            prompt, image, mask, os.path.abspath(args.object))
    else:
        import tools.codex_image_mask as tool_mod
        call = lambda prompt: tool_mod.codex_image_mask_tool(prompt, image, mask)
    base_prompt = args.prompt or (DEFAULT_PROMPT_REFERENCE if args.object else DEFAULT_PROMPT_MASK)

    # Each module has its own _CODEX_CHAT_MODEL, read by its _build_responses_payload at call time.
    if args.chat_model != "hermes":
        tool_mod._CODEX_CHAT_MODEL = args.chat_model
    print(f"Tool: {tool_mod.TOOL_NAME}", file=sys.stderr)

    if not args.no_size_hint:
        base_prompt = f"{base_prompt}\n\n{size_hint(mask)}"

    best = None  # (spill, result) with the lowest spill so far
    for attempt in range(1, max(args.attempts, 1) + 1):
        prompt = base_prompt
        if attempt > 1:
            scale = round(100 * 0.7 ** (attempt - 1))
            prompt += (f"\n\nA previous attempt drew the object too large and it extended outside the "
                       f"masked area. Draw it at about {scale}% of that size so it fits completely "
                       "inside the masked area.")
        print(f"--- Attempt {attempt}/{args.attempts}", file=sys.stderr)
        result = call(prompt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result.get("success"):
            continue
        spill = spill_percent(image, mask, result["image"])
        print(f"Spill outside mask: {spill}% (limit {args.max_spill}%)", file=sys.stderr)
        if best is None or spill < best[0]:
            best = (spill, result)
        if spill <= args.max_spill:
            break

    if best is None:
        return 1
    spill, result = best
    fits = spill <= args.max_spill
    if not fits:
        print(f"WARNING: every attempt spilled outside the mask; keeping the best one ({spill}%).",
              file=sys.stderr)
    if args.output:
        if args.composite:
            composite(image, mask, result["image"], args.output, args.feather)
        else:
            shutil.copyfile(result["image"], args.output)
        print(f"Saved: {args.output}")
    return 0 if fits else 2


if __name__ == "__main__":
    sys.exit(main())
