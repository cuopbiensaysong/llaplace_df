"""B20/B5 — the stage-1 latent cannot tell entities apart.

`encode_mu` runs `in_proj -> set transformer -> observation-weighted mean over N` with no
entity identity anywhere on the encode side, so `mu` is a function of the MULTISET of a
timestep's values: permuting the entities leaves it bitwise unchanged. The decoder's
`entity_emb` is then added to the same shared `z` for every entity, so it can only apply a
static per-entity offset. Measured consequence on real data (B20): every entity's
reconstruction correlates 1.000 with the cross-entity mean of the targets while 5.9 % of the
cross-entity variation survives, and Gate 0 sits exactly at the panel's own shared-variance
ceiling on all five cells -- a property of the dataset, with no model in it.

`entity_encode=True` tags the encoder input, so entities occupy separable subspaces of the
pooled sum. `mu` keeps its shape [B, T, C], so stage 3's `data_dim` is untouched.
"""

import pytest
import torch

from llapdiffusion.configs.config_utils import clone_config, refresh_artifact_paths
from llapdiffusion.latent_space.latent_vae import LatentVAE

B, T, N, C = 2, 6, 4, 8


def _vae(entity_encode: bool) -> LatentVAE:
    torch.manual_seed(0)
    return LatentVAE(
        seq_len=T,
        latent_dim=16,
        latent_channel=C,
        enc_layers=1,
        dec_layers=1,
        enc_heads=2,
        dec_heads=2,
        enc_ff=32,
        dec_ff=32,
        input_dim=2,
        output_dim=1,
        dropout=0.0,
        num_entities=N,
        entity_conditioned=True,
        entity_encode=entity_encode,
    ).eval()


def _tokens(values: torch.Tensor) -> torch.Tensor:
    """[B,T,N] values -> [B,T,N,2] tokens = [value * obs, obs], everything observed."""
    return torch.stack([values, torch.ones_like(values)], dim=-1)


def test_entity_encode_requires_num_entities():
    with pytest.raises(ValueError, match="entity_encode"):
        LatentVAE(seq_len=T, latent_dim=8, latent_channel=C, entity_encode=True)


@pytest.mark.parametrize("entity_encode, invariant", [(False, True), (True, False)])
def test_permuting_entities_changes_the_latent_only_once_tagged(entity_encode, invariant):
    """The defect, stated exactly: with no encoder-side identity the pooled latent is
    permutation-INVARIANT, so 'station 1 was warm and station 2 cold' and its swap are the
    same code -- the per-entity deviation is not representable, whatever the decoder does."""
    model = _vae(entity_encode)
    torch.manual_seed(1)
    values = torch.randn(B, T, N)
    perm = torch.tensor([2, 0, 3, 1])
    pad = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        mu, _ = model.encode_mu(_tokens(values), pad)
        mu_perm, _ = model.encode_mu(_tokens(values[:, :, perm]), pad)

    assert mu.shape == (B, T, C), "the latent must keep its shape: stage 3's data_dim"
    if invariant:
        torch.testing.assert_close(mu, mu_perm, rtol=0, atol=1e-6)
    else:
        assert (mu - mu_perm).abs().max() > 1e-4


def test_tagged_encoder_can_represent_per_entity_deviations():
    """A sufficient-capacity check on the mechanism, with no training: fit the encoder+decoder
    to a panel of shared signal + per-entity offsets and ask how much of the CROSS-ENTITY
    variation survives the round trip. That is B20's own metric (0.059 on noaa_uk)."""
    torch.manual_seed(3)
    shared = torch.randn(64, T, 1)
    offsets = torch.randn(1, 1, N) * 1.5           # static per entity, the easiest case
    values = shared + offsets                      # [64, T, N]
    pad = torch.zeros(values.shape[0], N, dtype=torch.bool)
    x_tok = _tokens(values)
    target = values.unsqueeze(-1)

    def retained(entity_encode: bool) -> float:
        model = _vae(entity_encode).train()
        opt = torch.optim.Adam(model.parameters(), lr=5e-2)
        for _ in range(300):
            opt.zero_grad()
            mu, _ = model.encode_mu(x_tok, pad)
            recon = model.decode_mu(mu, pad)
            loss = (recon - target).pow(2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode_mu(x_tok, pad)
            recon = model.decode_mu(mu, pad).squeeze(-1)
        # cross-entity std retained, averaged over windows and time
        return float((recon.std(dim=-1).mean() / values.std(dim=-1).mean()).item())

    assert retained(True) > 0.5


def test_the_two_architectures_cannot_share_an_artifact(tmp_path):
    legacy, tagged = _vae(False), _vae(True)
    torch.save({"model": legacy.state_dict()}, tmp_path / "vae.pt")
    payload = torch.load(tmp_path / "vae.pt", map_location="cpu", weights_only=False)

    legacy.load_state_dict(payload["model"])
    with pytest.raises(RuntimeError):
        tagged.load_state_dict(payload["model"])

    cfg = clone_config()
    cfg.VAE_DIR, cfg.PRED, cfg.VAE_LATENT_CHANNELS = "/tmp/vae", 168, 16
    cfg.VAE_ENTITY_CONDITION, cfg.VAE_ENTITY_ENCODE = True, False
    refresh_artifact_paths(cfg)
    assert cfg.VAE_CKPT.endswith("pred-168_ch-16_entity_elbo.pt")
    cfg.VAE_ENTITY_ENCODE = True
    refresh_artifact_paths(cfg)
    assert cfg.VAE_CKPT.endswith("pred-168_ch-16_entity_entenc_elbo.pt")
