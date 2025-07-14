import torch
import torch.optim as optim
import torch.nn as nn

from src.trainers.base_trainer import BaseTrainer
from src.models import ProjectedGANGenerator, ProjectedGANDiscriminator, FeatureExtractor
from src.utils import toggle_grad, denormalize_image
from src.losses.adversarial import generator_loss_bce, discriminator_loss_bce
from src.trainers.base_trainer import BaseTrainer

class ProjectedGANTrainer(BaseTrainer):
    def _init_models(self):
        self.G = ProjectedGANGenerator(self.config).to(self.device)
        self.D = ProjectedGANDiscriminator(self.config).to(self.device)
        self.E = None
        self.sp_latent_encoder = None
        self.feature_extractor = FeatureExtractor(
            model_name=self.config.model.projectedgan_feature_extractor_name,
            layers_to_extract=self.config.model.projectedgan_feature_layers_to_extract,
            pretrained=True,
            requires_grad=False
        ).to(self.device).eval()

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
        self.loss_fn_g_adv = generator_loss_bce
        self.loss_fn_d_adv = discriminator_loss_bce
        self.loss_fn_g_feat_match = nn.MSELoss()

    def _train_d(self, real_images, **kwargs):
        toggle_grad(self.D, True)
        self.optimizer_D.zero_grad()

        d_real_logits = self.D(real_images)

        z_dim_to_use = self.config.model.stylegan2_z_dim # ProjectedGAN uses StyleGAN2's z_dim
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

        z_dim_to_use_g = self.config.model.stylegan2_z_dim
        z_noise_g = torch.randn(real_images.size(0), z_dim_to_use_g, device=self.device)

        fake_images_for_g = self.G(z_noise_g)
        d_fake_logits_for_g = self.D(fake_images_for_g)

        lossG_adv = self.loss_fn_g_adv(d_fake_logits_for_g)
        logs = {"Loss_G_Adv": lossG_adv.item()}
        lossG = lossG_adv

        real_01 = denormalize_image(real_images)
        fake_01 = denormalize_image(fake_images_for_g)
        real_feats_dict = self.feature_extractor(real_01)
        fake_feats_dict = self.feature_extractor(fake_01)

        lossG_feat = 0.0
        for key in real_feats_dict:
            lossG_feat += self.loss_fn_g_feat_match(fake_feats_dict[key], real_feats_dict[key].detach())
        lossG += self.config.model.projectedgan_feature_matching_loss_weight * lossG_feat
        logs["Loss_G_FeatMatch"] = lossG_feat.item() if isinstance(lossG_feat, torch.Tensor) else lossG_feat

        lossG.backward()
        self.optimizer_G.step()
        logs["Loss_G_Total"] = lossG.item()

        toggle_grad(self.G, False)
        return logs
