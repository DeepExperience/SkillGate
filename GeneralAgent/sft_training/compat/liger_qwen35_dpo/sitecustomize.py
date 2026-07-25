"""Compatibility alias for Liger 0.8.0's Qwen3.5 non-fused path.

Liger 0.8.0's Qwen3.5 monkey patch imports ``liger_cross_entropy`` from
``transformers.cross_entropy``, while the same wheel defines that functional
API in ``transformers.functional``.  DPO requests logits and therefore selects
this non-fused branch.  Expose the wheel's own function at the path its own
monkey patch expects; no implementation or numerical behavior is changed.

This directory is prepended to PYTHONPATH only by the SkillGate DPO launcher.
"""

from liger_kernel.transformers import cross_entropy
from liger_kernel.transformers.functional import liger_cross_entropy
from transformers.utils import import_utils as transformers_import_utils


if not hasattr(cross_entropy, "liger_cross_entropy"):
    cross_entropy.liger_cross_entropy = liger_cross_entropy


# TRL 0.24 expects Transformers' private package probe to return a boolean
# unless ``return_version=True``.  The Transformers 5.8 development build in
# this frozen environment returns ``(available, version)`` in both cases, so a
# missing optional package becomes a truthy tuple.  Prime TRL's cached optional
# dependency flags under the legacy return contract, then immediately restore
# Transformers' function.  This remains local to the SkillGate DPO process.
_original_is_package_available = transformers_import_utils._is_package_available


def _trl_compatible_is_package_available(name: str, return_version: bool = False):
    result = _original_is_package_available(name, return_version=return_version)
    if return_version or not isinstance(result, tuple):
        return result
    return result[0]


transformers_import_utils._is_package_available = _trl_compatible_is_package_available
try:
    import trl.import_utils  # noqa: F401  # cache dependency flags under the compatibility contract
finally:
    transformers_import_utils._is_package_available = _original_is_package_available


from dpo_completion_logits import install as _install_completion_logits


_install_completion_logits()
