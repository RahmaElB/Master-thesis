"""
Thin wrapper around wandb so the rest of the code doesn't need to care whether
--use_wandb was passed, or whether wandb is even installed on the cluster.

"""

import os


class WandbLogger:
    def __init__(self, enabled: bool, project: str, run_name: str, config: dict):
        self.enabled = enabled
        self._wandb = None

        if not enabled:
            return

        try:
            import wandb
            self._wandb = wandb
            self._wandb.init(project=project, name=run_name, config=config)
        except Exception as e:  # pragma: no cover - defensive, cluster may lack wandb/network
            print(f"[wandb] could not initialize ({e}); continuing without wandb logging.")
            self.enabled = False
            self._wandb = None

    def log(self, metrics: dict, step: int = None) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
