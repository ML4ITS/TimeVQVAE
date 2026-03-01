import pytest
import torch

import timevqvae.vqvae as vqvae_mod


class FakeEncoder(torch.nn.Module):
    calls = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.kind = args[5]
        FakeEncoder.calls.append((args, kwargs))

    def forward(self, x):
        return x + (10.0 if self.kind == "lf" else 20.0)


class FakeDecoder(torch.nn.Module):
    calls = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.kind = args[6]
        FakeDecoder.calls.append((args, kwargs))

    def forward(self, x):
        return x + (1000.0 if self.kind == "lf" else 2000.0)


class FakeVectorQuantize(torch.nn.Module):
    calls = []

    def __init__(self, dim, codebook_size, **kwargs):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.kwargs = kwargs
        FakeVectorQuantize.calls.append((dim, codebook_size, kwargs))


@pytest.fixture
def patched_vqvae(monkeypatch):
    FakeEncoder.calls.clear()
    FakeDecoder.calls.clear()
    FakeVectorQuantize.calls.clear()

    downsample_calls = []

    def fake_compute_downsample_rate(input_length, n_fft, width):
        downsample_calls.append((input_length, n_fft, width))
        return width + 1

    def fake_quantize(z, vq_model):
        z_q = z + 100.0
        vq_loss = {
            "loss": torch.tensor(float(vq_model.codebook_size)),
            "commit_loss": torch.tensor(0.5),
        }
        perplexity = torch.tensor(float(vq_model.codebook_size) / 10.0)
        return z_q, None, vq_loss, perplexity

    monkeypatch.setattr(vqvae_mod, "VQVAEEncoder", FakeEncoder)
    monkeypatch.setattr(vqvae_mod, "VQVAEDecoder", FakeDecoder)
    monkeypatch.setattr(vqvae_mod, "VectorQuantize", FakeVectorQuantize)
    monkeypatch.setattr(vqvae_mod, "compute_downsample_rate", fake_compute_downsample_rate)
    monkeypatch.setattr(vqvae_mod, "quantize", fake_quantize)

    return downsample_calls


def test_init_uses_explicit_parameters_and_constructs_submodules(patched_vqvae):
    model = vqvae_mod.VQVAE(
        in_channels=3,
        input_length=64,
        n_fft=16,
        init_dim=6,
        hid_dim=32,
        downsampled_width_l=5,
        downsampled_width_h=9,
        encoder_n_resnet_blocks=4,
        decoder_n_resnet_blocks=7,
        codebook_size_l=128,
        codebook_size_h=256,
        kmeans_init=False,
        codebook_dim=12,
    )

    assert patched_vqvae == [(64, 16, 5), (64, 16, 9)]
    assert model.input_length == 64
    assert model.n_fft == 16

    enc_l_args, _ = FakeEncoder.calls[0]
    enc_h_args, _ = FakeEncoder.calls[1]
    assert enc_l_args[:6] == (6, 32, 6, 6, 4, "lf")
    assert enc_h_args[:6] == (6, 32, 6, 10, 4, "hf")

    dec_l_args, _ = FakeDecoder.calls[0]
    dec_h_args, _ = FakeDecoder.calls[1]
    assert dec_l_args[:7] == (6, 32, 6, 6, 7, 64, "lf")
    assert dec_h_args[:7] == (6, 32, 6, 10, 7, 64, "hf")

    assert FakeVectorQuantize.calls == [
        (32, 128, {"kmeans_init": False, "codebook_dim": 12}),
        (32, 256, {"kmeans_init": False, "codebook_dim": 12}),
    ]


def test_compute_frequency_targets_builds_lf_and_hf_time_targets(monkeypatch, patched_vqvae):
    model = vqvae_mod.VQVAE(in_channels=1, input_length=8)
    x = torch.randn(2, 3, 10)

    call_log = []

    def fake_time_to_timefreq(inp, n_fft, in_channels):
        call_log.append(("time_to_timefreq", inp.shape, n_fft, in_channels))
        return torch.zeros(inp.shape[0], inp.shape[1], 4, 5)

    def fake_zero_pad_high_freq(xf):
        call_log.append(("zero_pad_high_freq", xf.shape))
        return torch.ones_like(xf)

    def fake_zero_pad_low_freq(xf):
        call_log.append(("zero_pad_low_freq", xf.shape))
        return torch.full_like(xf, 2.0)

    def fake_timefreq_to_time(xf, n_fft, in_channels):
        tag = int(xf[0, 0, 0, 0].item())
        call_log.append(("timefreq_to_time", xf.shape, n_fft, in_channels, tag))
        out_val = 3.0 if tag == 1 else 7.0
        return torch.full((xf.shape[0], xf.shape[1], 3), out_val)

    monkeypatch.setattr(vqvae_mod, "time_to_timefreq", fake_time_to_timefreq)
    monkeypatch.setattr(vqvae_mod, "zero_pad_high_freq", fake_zero_pad_high_freq)
    monkeypatch.setattr(vqvae_mod, "zero_pad_low_freq", fake_zero_pad_low_freq)
    monkeypatch.setattr(vqvae_mod, "timefreq_to_time", fake_timefreq_to_time)

    x_l, x_h = model._compute_frequency_targets(x)

    assert x_l.shape == (2, 3, 8)
    assert x_h.shape == (2, 3, 8)
    assert torch.allclose(x_l, torch.full_like(x_l, 3.0))
    assert torch.allclose(x_h, torch.full_like(x_h, 7.0))
    assert call_log[0] == ("time_to_timefreq", x.shape, model.n_fft, x.shape[1])


def test_process_lf_branch_uses_lf_modules(patched_vqvae):
    model = vqvae_mod.VQVAE(in_channels=2, input_length=16, codebook_size_l=111, codebook_size_h=222)
    x = torch.zeros(1, 2, 16)

    xhat_l, vq_loss_l, perplexity_l = model._process_lf_branch(x)

    expected = x + 10.0 + 100.0 + 1000.0
    assert torch.allclose(xhat_l, expected)
    assert torch.isclose(vq_loss_l["loss"], torch.tensor(111.0))
    assert torch.isclose(perplexity_l, torch.tensor(11.1))


def test_process_hf_branch_uses_hf_modules(patched_vqvae):
    model = vqvae_mod.VQVAE(in_channels=2, input_length=16, codebook_size_l=111, codebook_size_h=222)
    x = torch.zeros(1, 2, 16)

    xhat_h, vq_loss_h, perplexity_h = model._process_hf_branch(x)

    expected = x + 20.0 + 100.0 + 2000.0
    assert torch.allclose(xhat_h, expected)
    assert torch.isclose(vq_loss_h["loss"], torch.tensor(222.0))
    assert torch.isclose(perplexity_h, torch.tensor(22.2))


def test_compute_losses_returns_expected_schema_values(patched_vqvae):
    model = vqvae_mod.VQVAE(in_channels=1, input_length=4)

    x_l = torch.tensor([[[1.0, 2.0]]])
    x_h = torch.tensor([[[2.0, 4.0]]])
    xhat_l = torch.tensor([[[1.0, 4.0]]])
    xhat_h = torch.tensor([[[1.0, 1.0]]])
    vq_loss_l = {"loss": torch.tensor(1.0)}
    vq_loss_h = {"loss": torch.tensor(2.0)}
    perplexity_l = torch.tensor(1.5)
    perplexity_h = torch.tensor(2.5)

    recons_loss, vq_losses, perplexities = model._compute_losses(
        x_l,
        x_h,
        xhat_l,
        xhat_h,
        vq_loss_l,
        vq_loss_h,
        perplexity_l,
        perplexity_h,
    )

    assert torch.isclose(recons_loss["LF.time"], torch.tensor(2.0))
    assert torch.isclose(recons_loss["HF.time"], torch.tensor(2.0))
    assert vq_losses == {"LF": vq_loss_l, "HF": vq_loss_h}
    assert perplexities == {"LF": perplexity_l, "HF": perplexity_h}


def test_forward_returns_dataclass_with_reconstruction_and_metrics(monkeypatch, patched_vqvae):
    model = vqvae_mod.VQVAE(in_channels=1, input_length=5)
    x = torch.randn(2, 1, 5)

    x_l = torch.ones(2, 1, 5)
    x_h = torch.ones(2, 1, 5) * 2
    xhat_l = torch.ones(2, 1, 5) * 3
    xhat_h = torch.ones(2, 1, 5) * 4
    vq_loss_l = {"loss": torch.tensor(10.0)}
    vq_loss_h = {"loss": torch.tensor(20.0)}
    perplexity_l = torch.tensor(1.25)
    perplexity_h = torch.tensor(2.5)

    monkeypatch.setattr(model, "_compute_frequency_targets", lambda inp: (x_l, x_h))
    monkeypatch.setattr(model, "_process_lf_branch", lambda inp: (xhat_l, vq_loss_l, perplexity_l))
    monkeypatch.setattr(model, "_process_hf_branch", lambda inp: (xhat_h, vq_loss_h, perplexity_h))

    out = model.forward(x)

    assert isinstance(out, vqvae_mod.VQVAEOutput)
    assert torch.allclose(out.x_recon, xhat_l + xhat_h)
    assert set(out.recons_loss.keys()) == {"LF.time", "HF.time"}
    assert out.vq_losses == {"LF": vq_loss_l, "HF": vq_loss_h}
    assert out.perplexities == {"LF": perplexity_l, "HF": perplexity_h}
