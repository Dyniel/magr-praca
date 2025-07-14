import torch
import torch.optim as optim
import torch.nn.functional as F

from src.models import StyleGAN2Generator, StyleGAN2Discriminator
from src.utils import toggle_grad
from src.losses.adversarial import generator_loss_wgan, discriminator_loss_wgan, gradient_penalty
from src.augmentations import ADAManager
from src.trainers.base_trainer import BaseTrainer

class StyleGAN2Trainer(BaseTrainer):
    def _init_models(self):
        self.G = StyleGAN2Generator(self.config).to(self.device)
        self.D = StyleGAN2Discriminator(self.config).to(self.device)
        self.E = None
        self.sp_latent_encoder = None
        self.w_avg = None
        if self.config.model.stylegan2_use_truncation:
            # Placeholder for w_avg calculation
            pass
        if hasattr(self.config.model, 'stylegan2_ada_target_metric_val'):
            self.ada_manager = ADAManager(self.config.model, self.device)

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
        self.loss_fn_g_adv = generator_loss_wgan
        self.loss_fn_d_adv = discriminator_loss_wgan

    def _train_d(self, real_images, **kwargs):
        toggle_grad(self.D, True)
        is_accumulation_step = self.current_iteration % self.config.gradient_accumulation_steps != 0

        d_input_real_images = real_images
        if self.ada_manager:
            d_input_real_images = self.ada_manager.apply_augmentations(real_images)

        d_real_logits = self.D(d_input_real_images)

        z_dim_to_use = self.config.model.stylegan2_z_dim
        z_noise = torch.randn(real_images.size(0), z_dim_to_use, device=self.device)

        g_kwargs = {
            'style_mix_prob': getattr(self.config.model, 'stylegan2_style_mix_prob', 0.9),
            'truncation_psi': None
        }

        with torch.no_grad():
            fake_images = self.G(z_noise, **g_kwargs)

        d_fake_logits = self.D(fake_images.detach())

        lossD_adv = self.loss_fn_d_adv(d_real_logits, d_fake_logits)
        gp = gradient_penalty(self.D, real_images, fake_images, self.device)
        lossD = lossD_adv + self.config.optimizer.lambda_gp * gp

        if torch.isnan(lossD):
            print("Warning: Total D loss is NaN. Skipping batch.")
            return {"Loss_D_Adv": "nan", "GP": "nan"}

        logs = {"Loss_D_Adv": lossD_adv.item(), "GP": gp.item()}

        lossD = lossD / self.config.gradient_accumulation_steps
        lossD.backward()

        if not is_accumulation_step:
            if any(torch.isnan(p.grad).any() for p in self.D.parameters() if p.grad is not None):
                print("Warning: NaN gradients in Discriminator. Skipping optimizer step.")
                self.optimizer_D.zero_grad()
            else:
                self.optimizer_D.step()
                self.optimizer_D.zero_grad()

        logs["Loss_D_Total"] = lossD.item() * self.config.gradient_accumulation_steps
        toggle_grad(self.D, False)
        return logs

    def _train_g(self, real_images, **kwargs):
        toggle_grad(self.G, True)
        is_accumulation_step = self.current_iteration % self.config.gradient_accumulation_steps != 0

        z_dim_to_use_g = self.config.model.stylegan2_z_dim
        z_noise_g = torch.randn(real_images.size(0), z_dim_to_use_g, device=self.device)

        g_kwargs_g = {
            'style_mix_prob': getattr(self.config.model, 'stylegan2_style_mix_prob', 0.9)
        }

        fake_images_for_g = self.G(z_noise_g, **g_kwargs_g)

        if self.ada_manager:
            fake_images_for_g_aug = self.ada_manager.apply_augmentations(fake_images_for_g)
        else:
            fake_images_for_g_aug = fake_images_for_g

        d_fake_logits_for_g = self.D(fake_images_for_g_aug)
        lossG_adv = self.loss_fn_g_adv(d_fake_logits_for_g)

        if torch.isnan(lossG_adv):
            print("Warning: Adversarial G loss is NaN. Skipping batch.")
            return {"Loss_G_Adv": "nan"}

        logs = {"Loss_G_Adv": lossG_adv.item()}
        lossG = lossG_adv

        lossG = lossG / self.config.gradient_accumulation_steps
        lossG.backward()

        if not is_accumulation_step:
            if any(torch.isnan(p.grad).any() for p in self.G.parameters() if p.grad is not None):
                print("Warning: NaN gradients in Generator. Skipping optimizer step.")
                self.optimizer_G.zero_grad()
            else:
                self.optimizer_G.step()
                self.optimizer_G.zero_grad()

        logs["Loss_G_Total"] = lossG.item() * self.config.gradient_accumulation_steps
        toggle_grad(self.G, False)
        return logs
