import os
import torch
import dataclasses
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils
import shutil
from PIL import Image
from torchvision import transforms

from src.models import (
    Generator as GeneratorGan5, Discriminator as DiscriminatorGan5,
    GraphEncoderGAT, GeneratorCNN, DiscriminatorCNN
)
from src.data_loader import get_dataloader # Assumes updated version with data_split
from src.utils import (
    PYG_AVAILABLE,
    save_checkpoint, load_checkpoint, setup_wandb, log_to_wandb,
    denormalize_image
)

if PYG_AVAILABLE:
    from torch_geometric.data import Batch as PyGBatch
    from torch_geometric.data import Data as PyGData
else:
    PyGBatch = None
    PyGData = None

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except ImportError:
    calculate_fid_given_paths = None

class Trainer:
    def __init__(self, config):
        print("Initializing Trainer (from src.trainer)...")
        self.config = config

        if calculate_fid_given_paths is None and getattr(self.config, 'enable_fid_calculation', False):
            print("pytorch-fid not found. FID calculation will be disabled.")
            if hasattr(self.config, 'enable_fid_calculation'):
                self.config.enable_fid_calculation = False

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # output_dir_run is expected to be set in config, e.g. by BaseConfig.__post_init__
        if not hasattr(config, 'output_dir_run') or not config.output_dir_run:
            raise ValueError("config.output_dir_run is not set. Ensure it's derived in your config (e.g. BaseConfig.__post_init__)")

        os.makedirs(self.config.output_dir_run, exist_ok=True)
        self.checkpoints_dir = os.path.join(self.config.output_dir_run, "checkpoints")
        self.samples_dir = os.path.join(self.config.output_dir_run, "samples")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model_architecture = config.model.architecture
        self.E = None
        self.optE = None

        if self.model_architecture == "gan5_gcn":
            self.G = GeneratorGan5(config).to(self.device)
            self.D = DiscriminatorGan5(config).to(self.device)
            self.optG = optim.Adam(self.G.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
            self.optD = optim.Adam(self.D.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
            self.loss_fn_d = self._loss_d_gan5
            self.loss_fn_g = self._loss_g_gan5
            print("Initialized gan5_gcn models and optimizers.")
        elif self.model_architecture == "gan6_gat_cnn":
            self.E = GraphEncoderGAT(config).to(self.device)
            self.G = GeneratorCNN(config).to(self.device)
            self.D = DiscriminatorCNN(config).to(self.device)
            self.optE = optim.Adam(self.E.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
            self.optG = optim.Adam(self.G.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
            self.optD = optim.Adam(self.D.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
            self.loss_fn_d = self._loss_d_gan6
            self.loss_fn_g = self._loss_g_gan6
            print("Initialized gan6_gat_cnn models (E, G, D) and optimizers.")
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")

        self.train_dataloader = get_dataloader(config, data_split="train", shuffle=True, drop_last=True)
        if self.train_dataloader is None:
            raise ValueError("Training dataloader could not be created. Check dataset_path and data.")

        self.fixed_sample_batch = self._prepare_fixed_sample_batch()

        model_to_watch = [self.G, self.E] if self.model_architecture == "gan6_gat_cnn" and self.E else self.G
        # Ensure wandb_project_name is in config
        wandb_proj_name = getattr(config, "wandb_project_name", "default_gan_project")
        self.wandb_run = setup_wandb(config, model_to_watch, project_name=wandb_proj_name, watch_model=True)


        self.current_epoch = 0 # This is the epoch number we are about to start or resume
        self.current_step = 0 # Optimizer steps taken overall

        if hasattr(self.config, 'resume_checkpoint_path') and self.config.resume_checkpoint_path and os.path.exists(self.config.resume_checkpoint_path):
            self.load_training_checkpoint(self.config.resume_checkpoint_path)
        # else: current_epoch and current_step remain 0

    def load_training_checkpoint(self, checkpoint_path):
        optE_to_load = self.optE if self.model_architecture == "gan6_gat_cnn" else None
        model_e_to_load = self.E if self.model_architecture == "gan6_gat_cnn" else None

        # load_checkpoint returns the epoch number that was *completed*
        completed_epoch, optimizer_step_at_checkpoint = load_checkpoint(
            checkpoint_path, self.G, self.D, model_e_to_load,
            self.optG, self.optD, optE_to_load, self.device
        )
        self.current_epoch = completed_epoch + 1 # Start training from the next epoch
        self.current_step = optimizer_step_at_checkpoint # Resume optimizer steps
        print(f"Resumed from checkpoint. Last completed epoch: {completed_epoch}. Next epoch to run: {self.current_epoch}. Resuming optimizer step: {self.current_step}")


    def _prepare_fixed_sample_batch(self):
        try:
            # Use a fresh instance of the training dataloader for fixed batch
            # This ensures it's not affected by main dataloader's state if num_workers > 0
            # However, if dataset is very large, this re-initialization can be slow.
            # For fixed samples, usually shuffle=False is better for consistency too.
            temp_dataloader = get_dataloader(self.config, data_split="train", shuffle=False, drop_last=False) # Changed shuffle & drop_last
            if temp_dataloader is None:
                print("Warning: Could not create temp_dataloader for fixed_sample_batch (train split). Sample generation might fail.")
                return None

            raw_fixed_batch = None
            # Try to get a batch. If dataset is smaller than num_samples_to_log, this might iterate a few times.
            # It's simpler if fixed_sample_batch is just the first batch from this non-shuffled loader.
            for batch_data in temp_dataloader:
                raw_fixed_batch = batch_data
                break # Take the first batch

            del temp_dataloader

            if raw_fixed_batch is None:
                print("Warning: Training dataloader (for fixed batch) yielded no data.")
                return None

            num_samples = self.config.num_samples_to_log

            if self.model_architecture == "gan5_gcn":
                if not isinstance(raw_fixed_batch, dict) or not all(k in raw_fixed_batch for k in ["image", "segments", "adj"]):
                    print(f"Warning: Fixed sample batch for gan5_gcn incorrect format. Type: {type(raw_fixed_batch)}")
                    return None
                # Ensure we don't take more samples than available in the batch
                actual_num_samples = min(num_samples, raw_fixed_batch["image"].size(0))
                if actual_num_samples == 0: return None
                return {
                    "image": raw_fixed_batch["image"][:actual_num_samples].to(self.device),
                    "segments": raw_fixed_batch["segments"][:actual_num_samples].to(self.device),
                    "adj": raw_fixed_batch["adj"][:actual_num_samples].to(self.device)
                }
            elif self.model_architecture == "gan6_gat_cnn":
                real_images_tensor, graph_batch_pyg = None, None
                if isinstance(raw_fixed_batch, tuple) and len(raw_fixed_batch) == 2:
                    real_images_tensor, graph_batch_pyg = raw_fixed_batch
                elif isinstance(raw_fixed_batch, list) and len(raw_fixed_batch) == 2 and \
                     isinstance(raw_fixed_batch[0], torch.Tensor) and \
                     (PYG_AVAILABLE and isinstance(raw_fixed_batch[1], PyGBatch)):
                    print("DEBUG: _prepare_fixed_sample_batch received list, processing as tuple for gan6.")
                    real_images_tensor, graph_batch_pyg = raw_fixed_batch[0], raw_fixed_batch[1]
                else:
                    print(f"DEBUG: Fixed sample batch for gan6_gat_cnn type: {type(raw_fixed_batch)}")
                    return None

                if not PYG_AVAILABLE or PyGBatch is None or not isinstance(graph_batch_pyg, PyGBatch):
                    print(f"Error: graph_batch_pyg is not PyGBatch. Type: {type(graph_batch_pyg)}")
                    return None

                num_graphs_in_batch = graph_batch_pyg.num_graphs if hasattr(graph_batch_pyg, 'num_graphs') else 0
                actual_num_samples = min(num_samples, num_graphs_in_batch, real_images_tensor.size(0))

                if actual_num_samples == 0: print("Warning: actual_num_samples is 0 for fixed batch."); return None

                data_list = graph_batch_pyg.to_data_list()
                sliced_graph_batch = PyGBatch.from_data_list(data_list[:actual_num_samples])
                return {
                    "image": real_images_tensor[:actual_num_samples].to(self.device),
                    "graph_batch": sliced_graph_batch.to(self.device)
                }
            return None
        except StopIteration:
            print("Warning: DataLoader exhausted for fixed sample batch (StopIteration)."); return None
        except Exception as e:
            print(f"Error in _prepare_fixed_sample_batch: {e}"); import traceback; traceback.print_exc(); return None

    def _loss_d_gan5(self, d_real_logits, d_fake_logits): return (F.softplus(-d_real_logits + d_fake_logits) + F.softplus(d_fake_logits - d_real_logits)).mean()
    def _loss_g_gan5(self, d_fake_for_g_logits): return -d_fake_for_g_logits.mean()
    def _loss_d_gan6(self, d_real_logits, d_fake_logits): return (self.bce_loss(d_real_logits, torch.ones_like(d_real_logits)) + self.bce_loss(d_fake_logits, torch.zeros_like(d_fake_logits))) * 0.5
    def _loss_g_gan6(self, d_fake_for_g_logits): return self.bce_loss(d_fake_for_g_logits, torch.ones_like(d_fake_for_g_logits))\n
    def _r1_gradient_penalty(self, real_images, d_real_logits):\n        grad_real = torch.autograd.grad(outputs=d_real_logits.sum(), inputs=real_images, create_graph=True, allow_unused=True)[0]\n        if grad_real is None: return torch.tensor(0.0, device=self.device)\n        return (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()\n\n    def train_epoch(self):\n        if self.model_architecture == \"gan5_gcn\": self.G.train(); self.D.train()\n        elif self.model_architecture == \"gan6_gat_cnn\": \n            if self.E: self.E.train()\n            self.G.train(); self.D.train()\n\n        loop = tqdm(self.train_dataloader, desc=f\"Epoch [{self.current_epoch}/{self.config.num_epochs}]\", leave=False)\n        epoch_stats = {k: 0.0 for k in ['d_loss','g_loss','d_adv','r1','d_real_l','d_fake_l']}\n        optimizer_steps_this_epoch = 0\n        grad_accum_steps = getattr(self.config, 'gradient_accumulation_steps', 1)\n\n        # Zero gradients once before the accumulation loop if grad_accum_steps > 1\n        if grad_accum_steps > 1:\n            self.optD.zero_grad(set_to_none=True)\n            self.optG.zero_grad(set_to_none=True)\n            if self.optE: self.optE.zero_grad(set_to_none=True)\n\n        micro_batch_stats_accum = {k: 0.0 for k in epoch_stats.keys()}\n        micro_batch_count = 0\n\n        for batch_idx, raw_batch_data in enumerate(loop):\n            if raw_batch_data is None: print(f\"Warning: Trainer None batch at idx {batch_idx}, skipping.\"); continue\n            \n            current_batch_size=0; real_images=None; graph_batch_pyg=None; segments_map=None; adj_matrix=None\n            # Data loading and validation\n            if self.model_architecture == \"gan5_gcn\":\n                if not (isinstance(raw_batch_data, dict) and all(k in raw_batch_data for k in [\"image\", \"segments\", \"adj\"])): continue\n                real_images=raw_batch_data[\"image\"].to(self.device); segments_map=raw_batch_data[\"segments\"].to(self.device); adj_matrix=raw_batch_data[\"adj\"].to(self.device); current_batch_size=real_images.size(0)\n            elif self.model_architecture == \"gan6_gat_cnn\":\n                if isinstance(raw_batch_data, tuple) and len(raw_batch_data)==2: real_images,graph_batch_pyg = raw_batch_data\n                elif isinstance(raw_batch_data, list) and len(raw_batch_data)==2 and isinstance(raw_batch_data[0],torch.Tensor) and (PYG_AVAILABLE and isinstance(raw_batch_data[1],PyGBatch)):\n                    print(\"DEBUG: train_epoch received list, processing as tuple for gan6.\"); real_images,graph_batch_pyg = raw_batch_data[0],raw_batch_data[1]\n                else: print(f\"DEBUG: Invalid batch type {type(raw_batch_data)} for gan6 (train_epoch).\"); continue\n                real_images=real_images.to(self.device)\n                if not PYG_AVAILABLE or not isinstance(graph_batch_pyg,PyGBatch):print(\"DEBUG: graph_batch_pyg not PyGBatch\"); continue\n                graph_batch_pyg=graph_batch_pyg.to(self.device); current_batch_size=real_images.size(0)\n            if current_batch_size == 0: continue\n\n            # --- D Update --- \n            if grad_accum_steps == 1: self.optD.zero_grad(set_to_none=True)\n            d_upd = getattr(self.config, 'd_updates_per_g_update', 1)\n            batch_d_loss, batch_d_adv, batch_r1, batch_d_real_l, batch_d_fake_l = 0,0,0,0,0\n            for _ in range(d_upd):\n                lossD_s, lossD_adv_s, r1_s, d_real_l_s, d_fake_l_s = torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0)\n                if self.model_architecture==\"gan5_gcn\":\n                    z=torch.randn(current_batch_size,self.config.model.z_dim,device=self.device); \n                    with torch.no_grad():fake_imgs=self.G(z,real_images,segments_map,adj_matrix)\n                    d_real_l_s=self.D(real_images); d_fake_l_s=self.D(fake_imgs.detach()); lossD_adv_s=self.loss_fn_d(d_real_l_s,d_fake_l_s)\n                    real_images.requires_grad_(True); d_real_gp=self.D(real_images);r1_s=self._r1_gradient_penalty(real_images,d_real_gp);real_images.requires_grad_(False)\n                    lossD_s=lossD_adv_s+self.config.r1_gamma*0.5*r1_s\n                elif self.model_architecture==\"gan6_gat_cnn\":\n                    real_images.requires_grad_(True);d_real_l_s=self.D(real_images)\n                    with torch.no_grad():z_g=self.E(graph_batch_pyg);fake_imgs=self.G(z_g,current_batch_size)\n                    d_fake_l_s=self.D(fake_imgs.detach());lossD_adv_s=self.loss_fn_d(d_real_l_s,d_fake_l_s)\n                    r1_s=self._r1_gradient_penalty(real_images,d_real_l_s);real_images.requires_grad_(False)\n                    lossD_s=lossD_adv_s+self.config.r1_gamma*0.5*r1_s\n                lossD_sc=lossD_s/(grad_accum_steps*d_upd);lossD_sc.backward()\n                batch_d_loss+=lossD_s.item();batch_d_adv+=lossD_adv_s.item();batch_r1+=r1_s.item();batch_d_real_l+=d_real_l_s.mean().item();batch_d_fake_l+=d_fake_l_s.mean().item()\n            for k,v in zip(['d_loss','d_adv','r1','d_real_l','d_fake_l'],[x/d_upd for x in [batch_d_loss,batch_d_adv,batch_r1,batch_d_real_l,batch_d_fake_l]]): micro_batch_stats_accum[k]+=v\n            \n            # --- G Update ---\n            if grad_accum_steps == 1: self.optG.zero_grad(set_to_none=True); \n            if self.optE and grad_accum_steps==1: self.optE.zero_grad(set_to_none=True)\n            lossG_s=torch.tensor(0.0)\n            if self.model_architecture==\"gan5_gcn\":\n                z=torch.randn(current_batch_size,self.config.model.z_dim,device=self.device);fake_imgs_g=self.G(z,real_images,segments_map,adj_matrix)\n                d_fake_g=self.D(fake_imgs_g);lossG_s=self.loss_fn_g(d_fake_g)\n            elif self.model_architecture==\"gan6_gat_cnn\":\n                z_g_g=self.E(graph_batch_pyg);fake_imgs_g=self.G(z_g_g,current_batch_size)\n                d_fake_g=self.D(fake_imgs_g);lossG_s=self.loss_fn_g(d_fake_g)\n            lossG_sc=lossG_s/grad_accum_steps;lossG_sc.backward()\n            micro_batch_stats_accum['g_loss']+=lossG_s.item()\n            micro_batch_count+=1\n\n            if (batch_idx+1)%grad_accum_steps==0 or (batch_idx+1)==len(self.train_dataloader):\n                self.optD.step();self.optG.step();\n                if self.optE:self.optE.step()\n                optimizer_steps_this_epoch+=1\n                avg_step_stats={k:v/micro_batch_count for k,v in micro_batch_stats_accum.items()}\n                for k,v in avg_step_stats.items(): epoch_stats[k]+=v # Accumulate for epoch average\n                if self.current_step%self.config.log_freq_step==0:\n                    log_d={f\"Train/{k.replace('_l','_Logits').replace('d_','D_').replace('g_','G_').replace('adv','Adv').replace('r1','R1_Penalty')}_step\":v for k,v in avg_step_stats.items()}\n                    log_d[\"Epoch\"]=self.current_epoch; log_to_wandb(self.wandb_run,log_d,step=self.current_step); loop.set_postfix(log_d)\n                self.current_step+=1\n                if grad_accum_steps > 1 and not ((batch_idx+1)==len(self.train_dataloader)):\n                    self.optD.zero_grad(set_to_none=True);self.optG.zero_grad(set_to_none=True);\n                    if self.optE:self.optE.zero_grad(set_to_none=True)\n                micro_batch_stats_accum={k:0.0 for k in epoch_stats.keys()};micro_batch_count=0\n        \n        if optimizer_steps_this_epoch > 0:\n            avg_epoch_stats={f\"Train/Epoch_Avg_{k.replace('_l','_Logits').replace('d_','D_').replace('g_','G_').replace('adv','Adv').replace('r1','R1_Penalty')}\":v/optimizer_steps_this_epoch for k,v in epoch_stats.items()}\n            avg_epoch_stats[\"Epoch_Num\"]=self.current_epoch;print(f\"Epoch {self.current_epoch} Avg Losses: D={avg_epoch_stats['Train/Epoch_Avg_D_Loss']:.4f}, G={avg_epoch_stats['Train/Epoch_Avg_G_Loss']:.4f}\")\n            log_to_wandb(self.wandb_run,avg_epoch_stats,step=self.current_step)\n\n    def _evaluate_on_split(self, data_split: str):\n        print(f\"Evaluating on {data_split} split...\")\n        eval_dataloader = get_dataloader(self.config, data_split=data_split, shuffle=False, drop_last=False)\n        if eval_dataloader is None: return {}\n        self.G.eval(); self.D.eval(); \n        if self.E: self.E.eval()\n        totals = {k:0.0 for k in ['d_loss','g_loss','d_adv','r1_eval','d_real_l','d_fake_l']}; num_b = 0\n        with torch.no_grad():\n            for raw_b in tqdm(eval_dataloader, desc=f\"Eval {data_split}\", leave=False):\n                if raw_b is None: continue\n                bs=0; r_imgs=None; g_batch=None; seg_map=None; adj_m=None; f_imgs=None\n                if self.model_architecture == \"gan5_gcn\":\n                    if not (isinstance(raw_b, dict) and all(k in raw_b for k in [\"image\", \"segments\", \"adj\"])): continue\n                    r_imgs=raw_b[\"image\"].to(self.device); seg_map=raw_b[\"segments\"].to(self.device); adj_m=raw_b[\"adj\"].to(self.device); bs=r_imgs.size(0)\n                    z=torch.randn(bs,self.config.model.z_dim,device=self.device); f_imgs=self.G(z,r_imgs,seg_map,adj_m)\n                elif self.model_architecture == \"gan6_gat_cnn\":\n                    if isinstance(raw_b,tuple) and len(raw_b)==2: r_imgs,g_batch=raw_b\n                    elif isinstance(raw_b,list) and len(raw_b)==2 and isinstance(raw_b[0],torch.Tensor) and (PYG_AVAILABLE and isinstance(raw_b[1],PyGBatch)): r_imgs,g_batch=raw_b[0],raw_b[1]\n                    else: continue\n                    r_imgs=r_imgs.to(self.device)\n                    if not PYG_AVAILABLE or not isinstance(g_batch,PyGBatch): continue\n                    g_batch=g_batch.to(self.device); bs=r_imgs.size(0)\n                    z_g=self.E(g_batch); f_imgs=self.G(z_g,bs)\n                if bs==0 or f_imgs is None: continue\n                d_r_l_val=self.D(r_imgs); d_f_l_val=self.D(f_imgs); lD_adv=self.loss_fn_d(d_r_l_val,d_f_l_val); r1_eval_b=torch.tensor(0.0) # R1 is 0 in eval\n                lD=lD_adv + self.config.r1_gamma*0.5*r1_eval_b; d_f_g=self.D(f_imgs); lG=self.loss_fn_g(d_f_g)\n                for k,v_val in zip(totals.keys(), [lD.item(),lG.item(),lD_adv.item(),r1_eval_b.item(),d_r_l_val.mean().item(),d_f_l_val.mean().item()]): totals[k]+=v_val\n                num_b+=1\n        if self.model_architecture == \"gan5_gcn\": self.G.train(); self.D.train()\n        elif self.model_architecture == \"gan6_gat_cnn\": \n            if self.E: self.E.train()\n            self.G.train(); self.D.train()\n        if num_b==0: return {}\n        key_map = {'d_loss':'Loss_D', 'g_loss':'Loss_G', 'd_adv':'Loss_D_Adv', 'r1_eval':'R1_Penalty_Eval', 'd_real_l':'D_Real_Logits_Mean', 'd_fake_l':'D_Fake_Logits_Mean'}\n        avg_m = { f\"{data_split}/{key_map[k]}\": v/num_b for k,v in totals.items() }\n        print(f\"Results for {data_split}: {avg_m}\"); return avg_m\n\n    def generate_samples(self, epoch, step):\n        if not self.fixed_sample_batch or not self.fixed_sample_batch.get(\"image\") or self.fixed_sample_batch[\"image\"].size(0) == 0:\n            print(\"Fixed sample batch not available or empty. Skipping sample generation.\"); return\n        if self.model_architecture == \"gan5_gcn\": self.G.eval()\n        elif self.model_architecture == \"gan6_gat_cnn\": \n            if self.E: self.E.eval()\n            self.G.eval()\n        with torch.no_grad():\n            gen_s, real_s = None, None; bs = self.fixed_sample_batch[\"image\"].size(0)\n            if self.model_architecture == \"gan5_gcn\":\n                z=torch.randn(bs,self.config.model.z_dim,device=self.device); gen_s=self.G(z,self.fixed_sample_batch[\"image\"],self.fixed_sample_batch[\"segments\"],self.fixed_sample_batch[\"adj\"])\n                real_s=self.fixed_sample_batch[\"image\"]\n            elif self.model_architecture == \"gan6_gat_cnn\":\n                if \"graph_batch\" not in self.fixed_sample_batch or self.E is None: return\n                z_g=self.E(self.fixed_sample_batch[\"graph_batch\"]); gen_s=self.G(z_g,bs); real_s=self.fixed_sample_batch[\"image\"]\n            if gen_s is not None and real_s is not None:\n                gen_d=denormalize_image(gen_s); real_d=denormalize_image(real_s)\n                vutils.save_image(vutils.make_grid(gen_d,nrow=int(np.sqrt(bs))), os.path.join(self.samples_dir,f\"fake_ep{epoch:04d}_st{step}.png\"))\n                if self.wandb_run:\n                    tbl=self.wandb_run.Table(columns=[\"Epoch\", \"Step\", \"ID\", \"Real\", \"Generated\"])\n                    for i in range(bs): tbl.add_data(epoch,step,i,self.wandb_run.Image(real_d[i].cpu()),self.wandb_run.Image(gen_d[i].cpu()))\n                    log_to_wandb(self.wandb_run, {f\"Sample_Comparisons_Epoch_{epoch}\": tbl}, step=step)\n        if self.model_architecture == \"gan5_gcn\": self.G.train()\n        elif self.model_architecture == \"gan6_gat_cnn\": \n            if self.E: self.E.train()\n            self.G.train()\n\n    def _save_images_for_fid(self, images_tensor, base_path, start_idx, num_to_save):\n        os.makedirs(base_path, exist_ok=True)\n        img_denorm = denormalize_image(images_tensor)\n        for i in range(num_to_save):\n            transforms.ToPILImage()(img_denorm[i].cpu()).save(os.path.join(base_path, f\"img_{start_idx + i}.png\"))\n\n    def calculate_fid_score(self, data_split_for_reals: str = \"test\"):\n        if not getattr(self.config,'enable_fid_calculation',False) or calculate_fid_given_paths is None: return float('nan')\n        print(f\"Calculating FID: reals from '{data_split_for_reals}'...\")\n        om_g,om_e,om_d = self.G.training,(self.E.training if self.E else None),self.D.training\n        self.G.eval(); self.D.eval(); \n        if self.E: self.E.eval()\n        path_r=os.path.join(self.config.output_dir_run,f\"fid_real_{data_split_for_reals}_ep{self.current_epoch}\")\n        path_f=os.path.join(self.config.output_dir_run,f\"fid_fake_ep{self.current_epoch}\")\n        if os.path.exists(path_r): shutil.rmtree(path_r)\n        if os.path.exists(path_f): shutil.rmtree(path_f)\n        os.makedirs(path_r,exist_ok=True); os.makedirs(path_f,exist_ok=True)\n        n_fid=self.config.fid_num_images; fid_dl=get_dataloader(self.config,data_split_for_reals,shuffle=False,drop_last=False)\n        if fid_dl is None: print(f\"FID Err: DL for '{data_split_for_reals}' None.\"); return float('nan')\n        gen_c,real_c=0,0\n        for raw_b in tqdm(fid_dl,desc=f\"Gen/Save FID imgs ({data_split_for_reals})\",leave=False):\n            if raw_b is None: continue\n            bs_r,bs_gen=0,0; r_b_imgs=None; g_b_pyg=None # Ensure g_b_pyg is defined for gan5 case too\n            if self.model_architecture==\"gan5_gcn\":\n                if not (isinstance(raw_b,dict) and \"image\" in raw_b): continue\n                r_b_imgs=raw_b[\"image\"]; bs_r=r_b_imgs.size(0); bs_gen=bs_r\n                seg_m,adj_m=raw_b[\"segments\"].to(self.device),raw_b[\"adj\"].to(self.device)\n            elif self.model_architecture==\"gan6_gat_cnn\":\n                if isinstance(raw_b,tuple) and len(raw_b)==2:tmp_r,tmp_g=raw_b\n                elif isinstance(raw_b,list) and len(raw_b)==2:tmp_r,tmp_g=raw_b[0],raw_b[1]\n                else: continue\n                r_b_imgs=tmp_r; bs_r=r_b_imgs.size(0)\n                if not PYG_AVAILABLE or not isinstance(tmp_g,PyGBatch): continue\n                g_b_pyg=tmp_g.to(self.device); bs_gen=g_b_pyg.num_graphs if hasattr(tmp_g, 'num_graphs') else 0\n            if bs_r==0 or bs_gen==0: continue\n            if real_c<n_fid: n_save_r=min(bs_r,n_fid-real_c); self._save_images_for_fid(r_b_imgs[:n_save_r],path_r,real_c,n_save_r); real_c+=n_save_r\n            if gen_c<n_fid:\n                n_gen_f=min(bs_gen,n_fid-gen_c); f_b=None\n                with torch.no_grad():\n                    if self.model_architecture==\"gan5_gcn\": z=torch.randn(n_gen_f,self.config.model.z_dim,device=self.device); f_b=self.G(z,r_b_imgs[:n_gen_f].to(self.device),seg_m[:n_gen_f],adj_m[:n_gen_f])\n                    elif self.model_architecture==\"gan6_gat_cnn\": s_g=g_b_pyg[:n_gen_f]; z_g=self.E(s_g); f_b=self.G(z_g,n_gen_f)\n                if f_b is not None: self._save_images_for_fid(f_b,path_f,gen_c,f_b.size(0)); gen_c+=f_b.size(0)\n            if real_c>=n_fid and gen_c>=n_fid: break\n        fid_v=float('nan'); min_imgs=getattr(self.config,\"fid_min_images\",10)\n        if gen_c>=min_imgs and real_c>=min_imgs:\n            try: fid_v=calculate_fid_given_paths([path_r,path_f],self.config.fid_batch_size,self.device,2048,getattr(self.config,'num_workers',0)); print(f\"FID ({data_split_for_reals},{gen_c}f,{real_c}r): {fid_v:.4f}\")\n            except Exception as e: print(f\"FID Error: {e}\")\n        else: print(f\"FID Skip: Need >={min_imgs}. Got {gen_c}f,{real_c}r.\")\n        self.G.train(om_g);self.D.train(om_d);\n        if self.E and om_e is not None:self.E.train(om_e)\n        return fid_v\n\n    def train(self):\n        print(\"Starting training (using main Trainer from src.trainer)...\")\n        eval_freq = getattr(self.config, \"eval_freq_epoch\", 1)\n        fid_freq = getattr(self.config, \"fid_freq_epoch\", 10)\n\n        for epoch_idx in range(self.current_epoch, self.config.num_epochs):\n            self.current_epoch = epoch_idx \n            self.train_epoch() \n\n            if epoch_idx % eval_freq == 0:\n                if hasattr(self.config, 'dataset_path_val') and self.config.dataset_path_val:\n                    val_metrics = self._evaluate_on_split(\"val\")\n                    if val_metrics: log_to_wandb(self.wandb_run, val_metrics, step=self.current_step)\n                if hasattr(self.config, 'dataset_path_test') and self.config.dataset_path_test:\n                    test_metrics = self._evaluate_on_split(\"test\")\n                    if test_metrics: log_to_wandb(self.wandb_run, test_metrics, step=self.current_step)\n\n            if epoch_idx % self.config.sample_freq_epoch == 0:\n                self.generate_samples(epoch_idx, self.current_step)\n\n            if getattr(self.config, 'enable_fid_calculation', False) and \\\n               (epoch_idx > 0 and (epoch_idx % fid_freq == 0 or epoch_idx == self.config.num_epochs - 1)):\n                fid_reals_split = \"test\" if hasattr(self.config, 'dataset_path_test') and self.config.dataset_path_test else \"train\"\n                temp_fid_dl = get_dataloader(self.config, data_split=fid_reals_split, shuffle=False, drop_last=False)\n                if temp_fid_dl is not None:\n                    fid_score = self.calculate_fid_score(data_split_for_reals=fid_reals_split)\n                    if not np.isnan(fid_score):\n                        log_to_wandb(self.wandb_run, {f\"FID_Score_{fid_reals_split.capitalize()}\": fid_score, \"Epoch_Num\": epoch_idx}, step=self.current_step)\n                else:\n                    print(f\"Skipping FID calculation: Dataloader for '{fid_reals_split}' could not be created.\")\n            \n            if epoch_idx % self.config.checkpoint_freq_epoch == 0 or epoch_idx == self.config.num_epochs - 1:\n                cp_data = {'epoch': epoch_idx, 'step': self.current_step,\n                           'G_state_dict': self.G.state_dict(), 'D_state_dict': self.D.state_dict(),\n                           'optG_state_dict': self.optG.state_dict(), 'optD_state_dict': self.optD.state_dict()}\n                if dataclasses.is_dataclass(self.config): cp_data['config'] = dataclasses.asdict(self.config)\n                else: cp_data['config'] = self.config \n                if self.model_architecture == \"gan6_gat_cnn\" and self.E and self.optE:\n                    cp_data['E_state_dict'] = self.E.state_dict(); cp_data['optE_state_dict'] = self.optE.state_dict()\n                save_checkpoint(cp_data, False, os.path.join(self.checkpoints_dir, f\"ckpt_ep{epoch_idx:04d}.pth.tar\"))\n                print(f\"Checkpoint saved for epoch {epoch_idx}\")\n\n        print(\"Training finished.\")\n        if self.wandb_run: self.wandb_run.finish()\n\n```
