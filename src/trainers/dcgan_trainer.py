import torch
import torch.optim as optim
import torch.nn.functional as F

from src.trainers.base_trainer import BaseTrainer
from src.models import DCGANGenerator, DCGANDiscriminator
from src.utils import toggle_grad
from src.losses.adversarial import generator_loss_bce, discriminator_loss_bce
from src.trainers.base_trainer import BaseTrainer


class DCGANTrainer(BaseTrainer):
    def _init_models(self):
        self.G = DCGANGenerator(self.config).to(self.device)
        self.D = DCGANDiscriminator(self.config).to(self.device)
        self.E = None
        self.sp_latent_encoder = None # DCGAN does not use this by default

    def _init_optimizers(self):
        g_params = list(self.G.parameters())
        self.optimizer_G = optim.Adam(
            g_params,
            lr=self.config.optimizer.g_lr,
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2)
        )
        self.optimizer_D = optim.Adam(
            self.D.parameters(),
            lr=self.config.optimizer.d_lr,
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2)
        )

    def _init_loss_functions(self):
        self.loss_fn_g_adv = lambda d_fake_logits: F.binary_cross_entropy_with_logits(d_fake_logits, torch.ones_like(d_fake_logits))
        self.loss_fn_d_adv = lambda d_real_logits, d_fake_logits: \
            F.binary_cross_entropy_with_logits(d_real_logits, torch.ones_like(d_real_logits)) + \
            F.binary_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))

    def _train_d(self, real_images, **kwargs):
        toggle_grad(self.D, True)
        self.optimizer_D.zero_grad()


        d_real_logits = self.D(real_images)

        z_dim_to_use = getattr(self.config.model, f"{self.model_architecture}_z_dim")
        z_noise = torch.randn(real_images.size(0), z_dim_to_use, device=self.device)

        with torch.no_grad():
            fake_images = self.G(z_noise)

        d_fake_logits = self.D(fake_images.detach())

        lossD = self.loss_fn_d_adv(d_real_logits, d_fake_logits)
        logs = {"Loss_D_Adv": lossD.item()}


        lossD.backward()
        self.optimizer_D.step()
        logs["Loss_D_Total"] = lossD.item()
        toggle_grad(self.D, False)
        return logs

    def _train_g(self, real_images, **kwargs):
        toggle_grad(self.G, True)
        self.optimizer_G.zero_grad()

        z_dim_to_use_g = getattr(self.config.model, f"{self.model_architecture}_z_dim")
        z_noise_g = torch.randn(real_images.size(0), z_dim_to_use_g, device=self.device)

        fake_images_for_g = self.G(z_noise_g)
        d_fake_logits_for_g = self.D(fake_images_for_g)

        lossG_adv = self.loss_fn_g_adv(d_fake_logits_for_g)
        logs = {"Loss_G_Adv": lossG_adv.item()}
        lossG = lossG_adv

        lossG.backward()
        self.optimizer_G.step()
        logs["Loss_G_Total"] = lossG.item()

        toggle_grad(self.G, False)
        return logs
