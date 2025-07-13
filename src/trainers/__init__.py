from .dcgan_trainer import DCGANTrainer
from .stylegan2_trainer import StyleGAN2Trainer
from .stylegan3_trainer import StyleGAN3Trainer
from .projectedgan_trainer import ProjectedGANTrainer

TRAINER_ZOO = {
    "dcgan": DCGANTrainer,
    "stylegan2": StyleGAN2Trainer,
    "stylegan3": StyleGAN3Trainer,
    "projected_gan": ProjectedGANTrainer,
}

def get_trainer(config):
    trainer_class = TRAINER_ZOO.get(config.model.architecture)
    if trainer_class:
        return trainer_class(config)
    else:
        raise ValueError(f"Unsupported model architecture: {config.model.architecture}")
