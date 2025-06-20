"""Centralised suppression of noisy third-party warnings.

This module is imported at package initialisation time and tweaks the Python
warnings filters **before** the heavyweight deep-learning libraries are pulled
in. The aim is to keep normal CLI usage as quiet as possible while still
allowing developers to re-enable warnings when needed (see *SPEAK_DEBUG*).

NOTE: We purposefully avoid blanket *catch-all* suppression - only clearly
harmless, extremely verbose warnings are ignored. New categories can be added
here over time if required.
"""

from __future__ import annotations

import warnings

# ---------------------------------------------------------------------------
# Warning filters
# ---------------------------------------------------------------------------

# 1. `pkg_resources` deprecation (triggered by *perth* dependency)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"perth\..*",
)

# 2. Diffusers LoRA deprecation spam
warnings.filterwarnings(
    "ignore",
    message=r"`?LoRACompatibleLinear`? is deprecated.*",
    category=FutureWarning,
    module=r"diffusers\..*",
)

# 3. PyTorch SDP kernel deprecation noise
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.backends\.cuda\.sdp_kernel\(\)` is deprecated.*",
    category=FutureWarning,
)

# 4. Transformers Whisper input rename
warnings.filterwarnings(
    "ignore",
    message=r"The input name `inputs` is deprecated.*",
    category=FutureWarning,
    module=r"transformers\..*",
)

# 5. Fallback: silence generic FutureWarnings from a few very chatty libs
for _mod in ("torch", "transformers", "diffusers"):
    warnings.filterwarnings("ignore", category=FutureWarning, module=rf"{_mod}\..*")


from transformers.utils import logging as hf_logging  # noqa

hf_logging.set_verbosity_error()

# Nothing to export - side-effects only
__all__: list[str] = []
