#!/home/rafal/.hermes/hermes-agent/venv/bin/python
"""Run Hermes' Codex masked image-edit tools directly, outside the agent.

With -r/--object: codex_image_mask_with_reference (main + mask + object reference image).
Without it:       codex_image_mask                (main + mask only).

Imports the tools from the Hermes source tree and calls them as-is, so they use the same
ChatGPT/Codex OAuth token Hermes uses (resolved by agent.auxiliary_client from ~/.hermes).

Usage:
    ./codex_mask_ref.py                                   # mask-only test run with image-edit-sunset PNGs
    ./codex_mask_ref.py -r ~/image-edit-sunset/object.png # reference test run
    ./codex_mask_ref.py -p "prompt" -i main.png -m mask.png [-r object.png] -o out.png
"""

import argparse
import json
import logging
import os
import shutil
import sys

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
                    help="Codex host model that calls image_generation (Hermes hardcodes gpt-5.5, "
                         "which this ChatGPT account rejects); pass 'hermes' to keep Hermes' value")
    args = ap.parse_args()

    if args.mode:
        os.environ["CODEX_IMAGE_MASK_MODE"] = args.mode
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Make Hermes packages (tools.*, agent.*) importable.
    sys.path.insert(0, HERMES_DIR)
    image, mask = os.path.abspath(args.image), os.path.abspath(args.mask)

    if args.object:
        import tools.codex_image_mask_with_reference as tool_mod
        prompt = args.prompt or DEFAULT_PROMPT_REFERENCE
        call = lambda: tool_mod.codex_image_mask_with_reference_tool(
            prompt, image, mask, os.path.abspath(args.object))
    else:
        import tools.codex_image_mask as tool_mod
        prompt = args.prompt or DEFAULT_PROMPT_MASK
        call = lambda: tool_mod.codex_image_mask_tool(prompt, image, mask)

    # Each module has its own _CODEX_CHAT_MODEL, read by its _build_responses_payload at call time.
    if args.chat_model != "hermes":
        tool_mod._CODEX_CHAT_MODEL = args.chat_model
    print(f"Tool: {tool_mod.TOOL_NAME}", file=sys.stderr)

    result = call()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not result.get("success"):
        return 1
    if args.output:
        shutil.copyfile(result["image"], args.output)
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
