"""Frozen foundation-model backbones for E9. Each loader returns (model, preprocess_fn, embed_dim) with
model already in eval mode and embed_fn(model, batch_float_0_1_224) -> (N, embed_dim) numpy array.

Availability checked 2026-10-29 (see research/experiments/E9_foundation_model_probes/backbone_check.log):
  dinov2_s    facebookresearch/dinov2 torch.hub, Apache-2.0, open -- always available
  dinobloom_s hf_hub:1aurent/vit_small_patch14_224.dinobloom (timm-compatible mirror), open
  reddino_s   hf_hub:Snarcy/RedDino-small (timm-compatible), open
  dinov3_s    facebook/dinov3-vits16-pretrain-lvd1689m (HF transformers), GATED -- requires a Hugging Face
              token with the licence accepted on the model page; falls back to skipped if unavailable
"""
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _normalize(x):
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def load_dinov2_s(device):
    # HF transformers, not torch.hub: torch.hub's repo-fork check hits the GitHub API (low anonymous rate
    # limit, hit on 2026-10-29) and provides no extra benefit here -- the HF weights are the same model.
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    def embed(x):
        with torch.no_grad():
            return m(pixel_values=_normalize(x).to(device)).pooler_output.cpu().numpy()
    return embed, m.config.hidden_size


def load_timm_hub(repo, device):
    import timm
    m = timm.create_model(f"hf_hub:{repo}", pretrained=True, num_classes=0).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
    def embed(x):
        with torch.no_grad():
            return m((x.to(device) - mean) / std).cpu().numpy()
    return embed, m.num_features


def load_dinov3_s(device):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    def embed(x):
        with torch.no_grad():
            out = m(pixel_values=_normalize(x).to(device))
            return out.pooler_output.cpu().numpy()
    return embed, m.config.hidden_size


REGISTRY = {
    "dinov2_s": lambda device: load_dinov2_s(device),
    "dinobloom_s": lambda device: load_timm_hub("1aurent/vit_small_patch14_224.dinobloom", device),
    "reddino_s": lambda device: load_timm_hub("Snarcy/RedDino-small", device),
    "dinov3_s": lambda device: load_dinov3_s(device),
}


def available_backbones(device, probe_only=True):
    """Try loading each backbone with a 1-image forward pass; returns {name: (embed_fn, dim)} for those
    that work, and prints a one-line reason for each that fails (e.g. DINOv3 gating)."""
    ok = {}
    for name, loader in REGISTRY.items():
        try:
            embed, dim = loader(device)
            if probe_only:
                embed(torch.zeros(1, 3, 224, 224))
            ok[name] = (embed, dim)
            print(f"{name}: OK, dim={dim}")
        except Exception as e:
            print(f"{name}: UNAVAILABLE ({type(e).__name__}: {str(e)[:150]})")
    return ok
