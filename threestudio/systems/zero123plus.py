import os
import random
import shutil
from dataclasses import dataclass, field

import math
import numpy
import torch
import einops
import torch_redstone as rst
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchmetrics import PearsonCorrCoef
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
from threestudio.models.guidance.zero123_diffusers import Zero123Pipeline


def to_rgb_image_13p(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.ones([rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8) * 127
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def to_rgb_image_13(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.ones([rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8) * 255
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def prepare_batch(elev, azim):
    THETA = torch.tensor([math.atan(32 / 2 / 35)])
    fovy = THETA * 2
    camera_distance = torch.tensor([2.7])
    height = 320
    width = 320
    elev = 90 - elev
    elevation = elev * math.pi / 180
    azimuth = azim * math.pi / 180
    camera_position: Float[Tensor, "1 3"] = torch.stack(
        [
            camera_distance * torch.cos(elevation) * torch.cos(azimuth),
            camera_distance * torch.cos(elevation) * torch.sin(azimuth),
            camera_distance * torch.sin(elevation),
        ],
        dim=-1,
    )

    center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
    up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

    light_position: Float[Tensor, "1 3"] = camera_position
    lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
    right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w: Float[Tensor, "1 3 4"] = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
        dim=-1,
    )
    
    directions_unit_focal = get_ray_directions(H=height, W=width, focal=1.0)
    focal_length = 0.5 * height / torch.tan(0.5 * fovy)

    directions: Float[Tensor, "1 H W 3"] = directions_unit_focal[None]
    directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length

    rays_o, rays_d = get_rays(
        directions, c2w, keepdim=True, noise_scale=2e-3
    )

    proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
        fovy, width / height, 0.1, 100.0
    )  # FIXME: hard-coded near and far
    mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(c2w, proj_mtx)

    return {
        "rays_o": rays_o.squeeze(0),
        "rays_d": rays_d.squeeze(0),
        "mvp_mtx": mvp_mtx.squeeze(0),
        "camera_positions": camera_position.squeeze(0),
        "light_positions": light_position.squeeze(0),
        "elevation": elev.squeeze(0),
        "azimuth": azim.squeeze(0),
        "camera_distances": camera_distance.squeeze(0),
        "height": height,
        "width": width,
    }


def zero123plus_guidance_prepare(self, image, guidance_scale=4):
    self.prepare()
    image_1 = self.feature_extractor_vae(images=image, return_tensors="pt").pixel_values
    image_2 = self.feature_extractor_clip(images=image, return_tensors="pt").pixel_values
    image = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
    image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
    cond_lat = self.encode_condition_image(image)
    if guidance_scale > 1:
        negative_lat = self.encode_condition_image(torch.zeros_like(image))
        cond_lat = torch.cat([negative_lat, cond_lat])
    encoded = self.vision_encoder(image_2, output_hidden_states=False)
    global_embeds = encoded.image_embeds
    global_embeds = global_embeds.unsqueeze(-2)
    
    encoder_hidden_states = self._encode_prompt(
        "",
        self.device,
        1,
        False
    )
    ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
    cak = dict(cond_lat=cond_lat)
    return torch.cat([encoder_hidden_states, encoder_hidden_states + global_embeds * ramp]), cak


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


def zero123plus_guidance_run(self, current, encoder_hidden_states, cak):
    # current: 6HWC
    c = einops.rearrange(current, "(s ph pw) h w c -> s c (ph h) (pw w)", s=1, ph=3, pw=2) * 2 - 1
    c = F.interpolate(
        c, (320 * 3, 320 * 2), mode="bilinear", align_corners=False
    )
    c = scale_image(c)
    c = self.encode_condition_image(c) * self.vae.config.scaling_factor
    latents = scale_latents(c)
    self.scheduler.set_timesteps(50)
    t = random.choice(self.scheduler.timesteps)
    with torch.no_grad():
        # add noise
        noise = torch.randn_like(latents)  # TODO: use torch generator
        latents_noisy = self.scheduler.add_noise(latents, noise, t.reshape(-1))
        # pred noise
        x_in = torch.cat([latents_noisy] * 2)
        t_in = t.reshape(-1).cuda()
        noise_pred = self.unet(
            x_in, t_in, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cak
        ).sample
    # perform guidance
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + 4.0 * (
        noise_pred_cond - noise_pred_uncond
    )
    x0 = self.scheduler.step(
        noise_pred, t, latents_noisy
    ).pred_original_sample
    w = 1
    grad = w * (x0 - latents)
    grad = torch.nan_to_num(grad)
    # clip grad for stable training?
    # if self.grad_clip_val is not None:
    #     grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

    # loss = SpecifyGradient.apply(latents, grad)
    # SpecifyGradient is not straghtforward, use a reparameterization trick instead
    target = (latents - grad).detach()
    # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
    loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum")
    return loss_sds


@threestudio.register("zero123-plus-system")
class Zero123Plus(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)
        refinement: bool = False
        ambient_ratio_min: float = 0.5
        cond_image_path: str = None

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # no prompt processor
        # self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.zero123 = Zero123Pipeline.from_pretrained(            
            # "bennyguo/zero123-diffusers",
            "bennyguo/zero123-xl-diffusers",
            variant="fp16_ema",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.zero123plus = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16
        ).to(self.device)
        self.zero123plus.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.zero123plus.scheduler.config, timestep_spacing='trailing'
        )

        self.cond_image = Image.open(self.cfg.cond_image_path)
        self.guidance_prep = zero123plus_guidance_prepare(self.zero123plus, self.cond_image)

        # visualize all training images
        all_images = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

        self.pearson = PearsonCorrCoef().to(self.device)

    def training_substep(self, batch, batch_idx, guidance: str):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "zero123"
        """
        if guidance == "ref":
            # bg_color = torch.rand_like(batch['rays_o'])
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
        elif guidance == "zero123":
            step = self.true_global_step
            if step == 0:
                self.buffer = []
            if step % 30 == 0:
                if len(self.buffer):
                    self.buffer.pop(0)
                while len(self.buffer) < 10:
                    self.zero123.to(self.device)
                    elev = random.randint(-10, 10)
                    azim = random.randint(-15, 15)
                    gf16 = lambda x: torch.tensor([x]).half().cuda()
                    gen = self.zero123(
                        to_rgb_image_13(self.cond_image).resize((256, 256)),
                        *gf16([elev, azim, 0]).reshape(3, 1),
                        num_inference_steps=30
                    ).images[0]
                    self.buffer.append((elev, azim, gen))
                self.zero123.to('cpu')
            # batch = batch["random_camera"]
            in_elev = 90 - batch['elevation']
            elev, azim, gen = random.choice(self.buffer)
            cams = [[30 + 60 * i, 60 if i % 2 == 0 else 105] for i in range(6)]
            old_batch = batch
            batch = rst.torch_to(rst.collate_support_object_proxy(
                [prepare_batch(*torch.tensor([elev, azim + delta]).reshape(-1, 1)) for delta, elev in cams]
            ), self.device)
            # breakpoint()
            ambient_ratio = (
                self.cfg.ambient_ratio_min
                + (1 - self.cfg.ambient_ratio_min) * random.random()
            )

        batch["bg_color"] = None
        batch["ambient_ratio"] = ambient_ratio

        out = self(batch)
        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "zero123"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        if guidance == "ref":
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]

            # color loss
            gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"] * (
                1 - gt_mask.float()
            )
            set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"]))

            # mask loss
            set_loss("mask", F.mse_loss(gt_mask.float(), out["opacity"]))

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
                valid_pred_depth = out["depth"][gt_mask].unsqueeze(1)
                with torch.no_grad():
                    A = torch.cat(
                        [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)]  # [B,]
                valid_pred_depth = out["depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # normal loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = (
                    1 - 2 * batch["ref_normal"][gt_mask.squeeze(-1)]
                )  # [B, 3]
                valid_pred_normal = (
                    2 * out["comp_normal"][gt_mask.squeeze(-1)] - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )
        elif guidance == "zero123":
            # zero123
            # guidance_out = self.guidance(
            #     out["comp_rgb"],
            #     **batch,
            #     rgb_as_latents=False,
            #     guidance_eval=guidance_eval,
            # )
            # claforte: TODO: rename the loss_terms keys
            set_loss("sds", zero123plus_guidance_run(self.zero123plus, out['comp_rgb'], *self.guidance_prep))

        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())

        if not self.cfg.refinement:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                set_loss(
                    "orient",
                    (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum()
                    / (out["opacity"] > 0).sum(),
                )

            if guidance != "ref" and self.C(self.cfg.loss.lambda_sparsity) > 0:
                set_loss("sparsity", (out["opacity"] ** 2 + 0.01).sqrt().mean())

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                set_loss(
                    "opaque", binary_cross_entropy(opacity_clamped, opacity_clamped)
                )
        else:
            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                set_loss("normal_consistency", out["mesh"].normal_consistency())
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                set_loss("laplacian_smoothness", out["mesh"].laplacian())

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        if self.cfg.freq.get("ref_or_zero123", "accumulate") == "accumulate":
            do_ref = True
            do_zero123 = True
        elif self.cfg.freq.get("ref_or_zero123", "accumulate") == "alternate":
            do_ref = (
                self.true_global_step < self.cfg.freq.ref_only_steps
                or self.true_global_step % self.cfg.freq.n_ref == 0
            )
            do_zero123 = not do_ref

        total_loss = 0.0
        if do_zero123:
            out = self.training_substep(batch, batch_idx, guidance="zero123")
            total_loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref")
            total_loss += out["loss"]

        self.log("train/loss", total_loss, prog_bar=True)

        # sch = self.lr_schedulers()
        # sch.step()

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0],
                        "kwargs": {},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
            name=f"validation_step_batchidx_{batch_idx}"
            if batch_idx in [0, 7, 15, 23, 29]
            else None,
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"
        self.save_img_sequence(
            filestem,
            filestem,
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="validation_epoch_end",
            step=self.true_global_step,
        )
        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
        )

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0],
                        "kwargs": {},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test")
        )
