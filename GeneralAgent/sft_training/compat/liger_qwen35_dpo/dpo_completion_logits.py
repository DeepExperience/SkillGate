"""Memory-safe Qwen3.5 DPO forward for the isolated SkillGate baseline.

The LLaMA-Factory DPO subclass bundled in this workspace predates TRL's
``logits_to_keep`` optimization.  It materializes logits for the full prompt,
which is prohibitive for SkillGate's roughly 18k-token prompts and 248k-token
vocabulary.  The preference loss only consumes labels in the assistant
completion, so retaining the logits beginning one token before the first
unmasked label is mathematically identical.

This patch is installed only from the SkillGate DPO compatibility
``sitecustomize`` directory; no default LLaMA-Factory path is changed.
"""

from __future__ import annotations

import torch


def install() -> None:
    """Install the completion-suffix forward on LLaMA-Factory's DPO trainer."""

    from llamafactory.train.dpo.trainer import CustomDPOTrainer
    from llamafactory.train.trainer_utils import get_batch_logps, nested_detach

    current = CustomDPOTrainer.concatenated_forward
    if getattr(current, "_skillgate_completion_logits", False):
        return

    def concatenated_forward(
        self: CustomDPOTrainer,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        is_ref_model: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)

        labels = batch.pop("labels")
        loss_mask = labels.ne(self.label_pad_token_id)
        nonzero = loss_mask.nonzero(as_tuple=True)
        if not nonzero[1].numel():
            raise ValueError("SkillGate DPO batch has no unmasked completion labels.")

        # A label at position t is predicted by the logit at t-1.  Keep that
        # predecessor plus the entire completion suffix.
        first_label_index = int(nonzero[1].min().item())
        logits_to_keep = labels.shape[1] - first_label_index + 1
        outputs = model(
            **batch,
            return_dict=True,
            use_cache=False,
            logits_to_keep=logits_to_keep,
        )
        if outputs.logits is None:
            raise RuntimeError("Qwen3.5 DPO forward returned no logits.")

        all_logits = outputs.logits.to(torch.float32)
        # Align labels with the suffix returned by ``logits_to_keep``.  The
        # existing helper then performs its normal one-token causal shift.
        labels = labels[:, -all_logits.shape[1] :]
        all_logps, valid_length = get_batch_logps(
            logits=all_logits,
            labels=labels,
            ld_alpha=(self.ld_alpha if not is_ref_model else None),
        )
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            chosen_logps_avg = chosen_logps
        else:
            chosen_logps_avg = chosen_logps / chosen_length

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_logps_avg": chosen_logps_avg,
        }

    concatenated_forward._skillgate_completion_logits = True  # type: ignore[attr-defined]
    CustomDPOTrainer.concatenated_forward = concatenated_forward

