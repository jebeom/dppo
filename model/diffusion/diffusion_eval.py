"""
For evaluating RL fine-tuned diffusion policy

Account for frozen base policy for early denoising steps and fine-tuned policy for later denoising steps

"""

import copy
import logging

import torch

log = logging.getLogger(__name__)

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.sampling import extract

from model.diffusion.diffusion import DiffusionModel, Sample
from model.diffusion.sampling import make_timesteps, extract


class DiffusionEval(DiffusionModel):
    def __init__(
        self,
        network_path,
        ft_denoising_steps,  # if running pre-trained model (not fine-tuned), set it to zero; if running fine-tuned model, need to specify the correct number of denoising steps fine-tuned, so that here it knows which model (base or ft) to use for each denoising step
        use_ddim=False,
        **kwargs,
    ):
        # do not let base class load model
        super().__init__(use_ddim=use_ddim, network_path=None, **kwargs)
        self.ft_denoising_steps = ft_denoising_steps
        checkpoint = torch.load(
            network_path, map_location=self.device, weights_only=True
        )  # 'network.mlp_mean...', 'actor.mlp_mean...', 'actor_ft.mlp_mean...'

        # Set up base model --- techncally not needed if all denoising steps are fine-tuned
        self.actor = self.network
        # import pdb;pdb.set_trace()
        try:
        # if True:
            base_weights = {
                key.split("actor.")[1]: checkpoint["model"][key]
                for key in checkpoint["model"]
                if "actor." in key
            }
            use_ft = True
            # Handle torch.compile wrapping (OptimizedModule) by unwrapping to _orig_mod
            actor_target = self.actor._orig_mod if hasattr(self.actor, "_orig_mod") else self.actor
            actor_target.load_state_dict(base_weights, strict=True)
        except Exception:
            assert ft_denoising_steps == 0, (
                "If no base policy weights are found, ft_denoising_steps must be 0"
            )
            # load ema weights to match DPPO finetune weights
            base_weights = {
                key.split("network.")[1]: checkpoint["ema"][key]
                for key in checkpoint["ema"]
                if "network." in key
            }
            use_ft = False
            logging.info("Actor weights not found. Using pre-trained weights!")
            actor_target = self.actor._orig_mod if hasattr(self.actor, "_orig_mod") else self.actor
            actor_target.load_state_dict(base_weights, strict=True)
        logging.info("Loaded base policy weights from %s", network_path)

        # Always set up fine-tuned model
        if use_ft:
            self.actor_ft = copy.deepcopy(self.network)
            ft_weights = {
                key.split("actor_ft.")[1]: checkpoint["model"][key]
                for key in checkpoint["model"]
                if "actor_ft." in key
            }
            actor_ft_target = (
                self.actor_ft._orig_mod if hasattr(self.actor_ft, "_orig_mod") else self.actor_ft
            )
            actor_ft_target.load_state_dict(ft_weights, strict=True)
            logging.info("Loaded fine-tuned policy weights from %s", network_path)
        self.use_guidance = False

    # override
    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        deterministic=False,
    ):
        noise = self.actor(x, t, cond=cond)

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε
            """
            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(cond).unsqueeze(1)  # B x 1 x (Da or 1)
            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)
            dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
            mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
            var = sigma**2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar

    def decode(self, cond: dict, steps: int = None, deterministic: bool = True):
        """
        Differentiable decoding of actions given condition and optional initial noise.

        Args:
            cond: dict containing at least key "state" of shape (B, To, Do).
                  Optionally include key "noise_action" of shape (B, Ta, Da) to specify
                  the initial latent x_T. If not provided, raises an error since this
                  class is used with controllable noise.
            steps: number of denoising steps to unroll. If None, uses the model's
                   configured number of steps (DDIM if enabled, otherwise DDPM).
            deterministic: if True and using DDIM, use eta=0 (already precomputed) and
                           no additional noise (fully deterministic trajectory).

        Returns:
            Tensor of shape (B, Ta, Da): the decoded action chunk.
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        batch_size = len(sample_data)

        # Initial latent x_T must be provided for controllable noise steering
        if "noise_action" not in cond:
            raise ValueError("cond must include 'noise_action' for controllable decoding")
        x = cond["noise_action"].to(device)

        # Determine timestep schedule
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        if steps is not None:
            # Take the first `steps` elements of the schedule
            # (the schedule is already ordered from T-1 -> 0)
            t_all = t_all[: int(steps)]

        # Unroll steps with gradients
        for i, t in enumerate(t_all):
            t_b = make_timesteps(batch_size, t, device)
            index_b = make_timesteps(batch_size, i, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                deterministic=deterministic,
            )

            # DDIM: std should be zero for deterministic sampling
            if self.use_ddim:
                std = torch.zeros_like(mean)
            else:
                # DDPM: make deterministic by zeroing std at final step; otherwise use clipped std
                if t == 0 or deterministic:
                    std = torch.zeros_like(mean)
                else:
                    std = torch.exp(0.5 * logvar).clamp_(min=1e-3)

            # Use zero noise for determinism; if std is zero, noise term is ignored
            noise = torch.zeros_like(x)
            x = mean + std * noise

            # Clamp only at the final step if configured
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )

        return x

