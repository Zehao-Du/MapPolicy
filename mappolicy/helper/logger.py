import os
# import uuid
import hydra
from termcolor import colored, cprint
import wandb
from omegaconf import OmegaConf

class Logger:
    """Logger class for printing messages with different colors."""

    @staticmethod
    def log(hint, *args, **kwargs):
        color = kwargs.pop("color", "blue")
        print(colored(f"[{hint}]", color), *args, **kwargs)

    @staticmethod
    def log_info(*args, **kwargs):
        Logger.log("INFO", *args, **kwargs, color="blue")

    @staticmethod
    def log_warning(*args, **kwargs):
        Logger.log("WARNING", *args, **kwargs, color="yellow")

    @staticmethod
    def log_error(*args, **kwargs):
        Logger.log("ERROR", *args, **kwargs, color="red")

    @staticmethod
    def log_ok(*args, **kwargs):
        Logger.log("OK", *args, **kwargs, color="green")

    @staticmethod
    def log_notice(*args, **kwargs):
        Logger.log("NOTICE", *args, **kwargs, color="magenta")

    @staticmethod
    def print_seperator(char="-", color="cyan"):
        cprint(char * os.get_terminal_size().columns, color)
        
class WandBLogger:
    def __init__(self, config, hyperparameters=None):
        wandb_cfg = config.get("wandb")
        if wandb_cfg is None:
            Logger.log_error("Config is missing 'wandb' section.")
            exit(1)
        # api_key = wandb_cfg.get("api_key")
        # email = wandb_cfg.get("email")
        # username = wandb_cfg.get("username")
        
        # os.environ["WANDB_API_KEY"] = api_key
        # os.environ["WANDB_USER_EMAIL"] = email
        # os.environ["WANDB_USERNAME"] = username
        
        self.run = wandb.init(
            project=wandb_cfg["project"],
            config=hyperparameters,
            group=wandb_cfg["group"],
            name=wandb_cfg["name"],
            notes=wandb_cfg["notes"],
            reinit=wandb_cfg["reinit"],
            mode=wandb_cfg["mode"],
            # id=uuid.uuid4().hex,
            dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)