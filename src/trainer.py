def _evaluate_on_split(self, data_split: str):
        """
        Evaluates the model on a given data split (e.g., "val" or "test").
        Calculates losses and other relevant metrics.
        """
        print(f"Evaluating on {data_split} split...")
        eval_dataloader = get_dataloader(self.config, data_split=data_split, shuffle=False, drop_last=False)

        if eval_dataloader is None:
            print(f"No dataloader for {data_split} split (path likely not configured). Skipping evaluation.")
            return {}

        self.G.eval()
        self.D.eval()
        if self.E: self.E.eval()
        if self.sp_latent_encoder: self.sp_latent_encoder.eval()

        total_d_loss = 0.0
        total_g_loss = 0.0
        total_d_loss_adv = 0.0
        total_r1_penalty = 0.0 # R1 is typically not computed/enforced during eval
        total_d_real_logits = 0.0
        total_d_fake_logits = 0.0
        total_g_feat_match_loss = 0.0 # For ProjectedGAN
        num_batches = 0

        with torch.no_grad():
            for raw_batch_data in tqdm(eval_dataloader, desc=f"Evaluating {data_split}"):
                lossD_batch = torch.tensor(0.0, device=self.device)
                lossG_batch = torch.tensor(0.0, device=self.device)
                lossD_adv_batch = torch.tensor(0.0, device=self.device)
                # r1_penalty_val = torch.tensor(0.0, device=self.device) # R1 not computed
                d_real_logits_mean_batch = torch.tensor(0.0, device=self.device)
                d_fake_logits_mean_batch = torch.tensor(0.0, device=self.device)
                lossG_feat_match_batch = torch.tensor(0.0, device=self.device)
                current_batch_size = 0

                # --- Prepare inputs for current batch (real images, segments, etc.) ---
                eval_real_images_gan_norm = None; eval_segments_map = None; eval_adj_matrix = None; eval_graph_batch_pyg = None
                if isinstance(raw_batch_data, dict) and "image" in raw_batch_data:
                    eval_real_images_gan_norm = raw_batch_data["image"].to(self.device)
                    if "segments" in raw_batch_data: eval_segments_map = raw_batch_data["segments"].to(self.device)
                    if "adj" in raw_batch_data: eval_adj_matrix = raw_batch_data["adj"].to(self.device)
                elif isinstance(raw_batch_data, tuple) and len(raw_batch_data) == 2:
                    eval_real_images_gan_norm, eval_graph_batch_pyg = raw_batch_data
                    eval_real_images_gan_norm = eval_real_images_gan_norm.to(self.device)
                    eval_graph_batch_pyg = eval_graph_batch_pyg.to(self.device)
                elif isinstance(raw_batch_data, torch.Tensor):
                    eval_real_images_gan_norm = raw_batch_data.to(self.device)

                if eval_real_images_gan_norm is None:
                    print(f"Warning: Could not extract real images for eval batch in {data_split} for arch {self.model_architecture}. Skipping batch.")
                    continue
                current_batch_size = eval_real_images_gan_norm.size(0)
                if current_batch_size == 0: continue

                # --- Prepare Superpixel Conditioning Tensors for G and D (if enabled) ---
                eval_spatial_map_g, eval_spatial_map_d, eval_z_superpixel_g = None, None, None
                g_spatial_active = getattr(self.config.model, f"{self.model_architecture}_g_spatial_cond", False)
                d_spatial_active = getattr(self.config.model, f"{self.model_architecture}_d_spatial_cond", False)
                g_latent_active = self.sp_latent_encoder is not None and \
                                  getattr(self.config.model, f"{self.model_architecture}_g_latent_cond", False)

                if self.config.model.use_superpixel_conditioning and eval_segments_map is not None and \
                   (g_spatial_active or d_spatial_active or g_latent_active):
                    eval_real_images_01 = denormalize_image(eval_real_images_gan_norm)
                    if g_spatial_active:
                        eval_spatial_map_g = generate_spatial_superpixel_map(
                            eval_segments_map, self.config.model.superpixel_spatial_map_channels_g,
                            self.config.image_size, self.config.num_superpixels, eval_real_images_01).to(self.device)
                        if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan", "dcgan"] and \
                           hasattr(self.config.model, "superpixel_spatial_map_channels_g") and self.config.model.superpixel_spatial_map_channels_g > 0 and \
                           eval_spatial_map_g is not None and eval_spatial_map_g.shape[-1] != 4:
                             eval_spatial_map_g = F.interpolate(eval_spatial_map_g, size=(4,4), mode='nearest')
                    if d_spatial_active: # For D, use map from real image's segments
                        eval_spatial_map_d = generate_spatial_superpixel_map(
                            eval_segments_map, self.config.model.superpixel_spatial_map_channels_d,
                            self.config.image_size, self.config.num_superpixels, eval_real_images_01).to(self.device)
                    if g_latent_active:
                        mean_sp_feats_eval = calculate_mean_superpixel_features(
                            eval_real_images_01, eval_segments_map,
                            self.sp_latent_encoder.num_superpixels, self.config.model.superpixel_feature_dim).to(self.device)
                        eval_z_superpixel_g = self.sp_latent_encoder(mean_sp_feats_eval)

                # --- Architecture-specific evaluation logic ---
                d_real_logits = self.D(eval_real_images_gan_norm, spatial_map_d=eval_spatial_map_d) # Common D call structure

                if self.model_architecture == "gan5_gcn":
                    z = torch.randn(current_batch_size, self.config.model.z_dim, device=self.device)
                    fake_images = self.G(z, eval_real_images_gan_norm, eval_segments_map, eval_adj_matrix)
                    d_fake_logits = self.D(fake_images) # gan5 D doesn't take spatial_map_d
                    lossD_adv_batch = self.loss_fn_d(d_real_logits, d_fake_logits)
                    lossG_batch = self.loss_fn_g(self.D(fake_images))
                elif self.model_architecture == "gan6_gat_cnn":
                    z_graph = self.E(eval_graph_batch_pyg)
                    if self.config.model.gan6_gat_cnn_use_null_graph_embedding: z_graph = torch.zeros_like(z_graph)
                    fake_images = self.G(z_graph, current_batch_size)
                    d_fake_logits = self.D(fake_images) # gan6 D doesn't take spatial_map_d
                    lossD_adv_batch = self.loss_fn_d(d_real_logits, d_fake_logits)
                    lossG_batch = self.loss_fn_g(self.D(fake_images))
                elif self.model_architecture == "dcgan":
                    z_noise = torch.randn(current_batch_size, self.config.model.z_dim, device=self.device)
                    fake_images = self.G(z_noise, spatial_map_g=eval_spatial_map_g, z_superpixel_g=eval_z_superpixel_g)
                    d_fake_logits = self.D(fake_images, spatial_map_d=eval_spatial_map_d)
                    lossD_adv_batch = self.loss_fn_d(d_real_logits, d_fake_logits)
                    lossG_batch = self.loss_fn_g(self.D(fake_images, spatial_map_d=eval_spatial_map_d))
                elif self.model_architecture == "stylegan2":
                    z_noise = torch.randn(current_batch_size, self.config.model.stylegan2_z_dim, device=self.device)
                    fake_images = self.G(z_noise, spatial_map_g=eval_spatial_map_g, z_superpixel_g=eval_z_superpixel_g, style_mix_prob=0)
                    d_fake_logits = self.D(fake_images, spatial_map_d=eval_spatial_map_d)
                    lossD_adv_batch = self.loss_fn_d_stylegan2(d_real_logits, d_fake_logits)
                    lossG_batch = self.loss_fn_g_stylegan2(self.D(fake_images, spatial_map_d=eval_spatial_map_d))
                elif self.model_architecture == "stylegan3":
                    z_noise = torch.randn(current_batch_size, self.config.model.stylegan3_z_dim, device=self.device)
                    fake_images = self.G(z_noise, spatial_map_g=eval_spatial_map_g, z_superpixel_g=eval_z_superpixel_g)
                    d_fake_logits = self.D(fake_images, spatial_map_d=eval_spatial_map_d)
                    lossD_adv_batch = self.loss_fn_d_stylegan3(d_real_logits, d_fake_logits)
                    lossG_batch = self.loss_fn_g_stylegan3(self.D(fake_images, spatial_map_d=eval_spatial_map_d))
                elif self.model_architecture == "projected_gan":
                    z_noise = torch.randn(current_batch_size, self.config.model.stylegan2_z_dim, device=self.device) # Uses SG2 z_dim
                    fake_images_gan_norm = self.G(z_noise, spatial_map_g=eval_spatial_map_g, z_superpixel_g=eval_z_superpixel_g, style_mix_prob=0)
                    d_fake_logits = self.D(fake_images_gan_norm, spatial_map_d=eval_spatial_map_d)
                    lossD_adv_batch = self.loss_fn_d_adv_projected(d_real_logits, d_fake_logits)

                    # G loss for ProjectedGAN also includes feature matching
                    lossG_adv_eval = self.loss_fn_g_adv_projected(self.D(fake_images_gan_norm, spatial_map_d=eval_spatial_map_d))
                    real_01 = denormalize_image(eval_real_images_gan_norm); fake_01 = denormalize_image(fake_images_gan_norm)
                    real_im_norm = self.imagenet_norm(real_01); fake_im_norm = self.imagenet_norm(fake_01)
                    real_feats = self.feature_extractor(real_im_norm); fake_feats = self.feature_extractor(fake_im_norm)
                    lossG_feat_match_batch = self.loss_fn_g_feat_match(real_feats, fake_feats)
                    lossG_batch = lossG_adv_eval + self.config.model.projectedgan_feature_matching_loss_weight * lossG_feat_match_batch
                    total_g_feat_match_loss += lossG_feat_match_batch.item()
                else:
                    print(f"Evaluation logic for {self.model_architecture} not fully defined. Setting dummy losses.")
                    lossD_adv_batch, lossG_batch = torch.tensor(0.0), torch.tensor(0.0)

                lossD_batch = lossD_adv_batch # R1 penalty not added in eval
                d_real_logits_mean_batch = d_real_logits.mean()
                d_fake_logits_mean_batch = d_fake_logits.mean()

                total_d_loss += lossD_batch.item()
                total_g_loss += lossG_batch.item()
                total_d_loss_adv += lossD_adv_batch.item()
                total_d_real_logits += d_real_logits_mean_batch.item()
                total_d_fake_logits += d_fake_logits_mean_batch.item()
                num_batches += 1

        self.G.train(); self.D.train()
        if self.E: self.E.train()
        if self.sp_latent_encoder: self.sp_latent_encoder.train()

        if num_batches == 0: return {}

        avg_metrics = {
            f"{data_split}/Loss_D": total_d_loss / num_batches,
            f"{data_split}/Loss_G": total_g_loss / num_batches,
            f"{data_split}/Loss_D_Adv": total_d_loss_adv / num_batches,
            f"{data_split}/D_Real_Logits_Mean": total_d_real_logits / num_batches,
            f"{data_split}/D_Fake_Logits_Mean": total_d_fake_logits / num_batches,
        }
        if self.model_architecture == "projected_gan":
            avg_metrics[f"{data_split}/Loss_G_FeatMatch"] = total_g_feat_match_loss / num_batches

        print(f"Evaluation results for {data_split}: {avg_metrics}")
        return avg_metrics
