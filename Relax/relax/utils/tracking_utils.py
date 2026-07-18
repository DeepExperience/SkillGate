# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import wandb

from relax.utils.logging_utils import get_logger
from relax.utils.metrics.adapters.apprise import _AppriseAdapter
from relax.utils.metrics.adapters.clearml import _ClearMLAdapter
from relax.utils.metrics.adapters.tensorboard import _TensorboardAdapter
from relax.utils.metrics.adapters.wandb import init_wandb_primary, init_wandb_secondary
from relax.utils.metrics.metrics_service_adapter import get_metrics_service_adapter, init_metrics_service_adapter


logger = get_logger(__name__)
_WARNED_WANDB_NOT_INITIALIZED = False


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        init_wandb_primary(args, **kwargs)
    else:
        init_wandb_secondary(args, **kwargs)

    # Initialize metrics service adapter if using new service-based logging
    if getattr(args, "use_metrics_service", False):
        init_metrics_service_adapter(args)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    # Check if using new metrics service
    if getattr(args, "use_metrics_service", False):
        adapter = get_metrics_service_adapter()
        if adapter:
            return adapter.log(metrics, step_key)
        raise ValueError("Warning: Metrics service adapter not initialized, falling back to direct logging")

    # Fall back to direct logging if metrics service is not used or not available
    if args.use_wandb:
        global _WARNED_WANDB_NOT_INITIALIZED
        if wandb.run is None and getattr(args, "wandb_run_id", None) is not None:
            init_wandb_secondary(args)
        if wandb.run is not None:
            wandb.log(metrics)
        elif not _WARNED_WANDB_NOT_INITIALIZED:
            logger.warning("Skipping wandb.log because no W&B run is initialized in this process.")
            _WARNED_WANDB_NOT_INITIALIZED = True

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    if args.use_clearml:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _ClearMLAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    if getattr(args, "notify_urls", None):
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _AppriseAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    return True


def flush_metrics(args, step: int):
    """Flush any buffered metrics in the metrics service adapter.

    This should be called at the end of each step to ensure all metrics are
    reported to the metrics service.
    """
    if getattr(args, "use_metrics_service", False):
        adapter = get_metrics_service_adapter()
        if adapter:
            return adapter.flush(step)
    return True


def log_direct(args, step: int, metrics: dict):
    """Direct logging to metrics service without step_key parsing.

    This is a simpler interface for cases where you already have the step
    separated from metrics.

    Args:
        args: Configuration arguments
        step: Step number
        metrics: Dictionary of metrics to log
    """
    if getattr(args, "use_metrics_service", False):
        adapter = get_metrics_service_adapter()
        if adapter:
            return adapter.direct_log(step, metrics)

    # Fall back to traditional logging
    metrics_with_step = metrics.copy()
    metrics_with_step["step"] = step
    return log(args, metrics_with_step, "step")
