"""Stop an HF Trainer cleanly on SIGTERM, which is what the gaming-pc `stop` task sends.

The Trainer does not save on SIGTERM by default. This callback saves a checkpoint at the end of the
current step and stops training. Jobs started by `run` have infra/gaming-pc on PYTHONPATH:

    from graceful import StopOnSigterm, last_checkpoint
    trainer = Trainer(..., callbacks=[StopOnSigterm()])
    trainer.train(resume_from_checkpoint=last_checkpoint(args.output_dir))
"""
import os, re, signal
from transformers import TrainerCallback


class StopOnSigterm(TrainerCallback):
    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGTERM, self._on_term)

    def _on_term(self, *_):
        print("SIGTERM: will checkpoint and stop after this step", flush=True)
        self.requested = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.requested:
            control.should_save = True
            control.should_training_stop = True
        return control


def last_checkpoint(output_dir):
    """The newest checkpoint-N folder in output_dir, or None to start fresh."""
    if not os.path.isdir(output_dir):
        return None
    steps = [int(m.group(1)) for d in os.listdir(output_dir) if (m := re.fullmatch(r"checkpoint-(\d+)", d))]
    return os.path.join(output_dir, f"checkpoint-{max(steps)}") if steps else None
