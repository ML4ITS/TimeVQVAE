from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import timevqvae.maskgit as maskgit_mod
from timevqvae.maskgit import PriorModelConfig


def test_gamma_func_modes_and_invalid():
    assert maskgit_mod._MaskingLogic.gamma_func("linear")(0.25) == pytest.approx(0.75)
    assert maskgit_mod._MaskingLogic.gamma_func("cosine")(0.0) == pytest.approx(1.0)
    assert maskgit_mod._MaskingLogic.gamma_func("square")(0.5) == pytest.approx(0.75)
    assert maskgit_mod._MaskingLogic.gamma_func("cubic")(0.5) == pytest.approx(0.875)

    with pytest.raises(NotImplementedError):
        maskgit_mod._MaskingLogic.gamma_func("unknown")


def test_randomly_mask_tokens_masks_expected_positions(monkeypatch):
    masking = maskgit_mod._MaskingLogic(
        lf_mask_token_id=99,
        hf_mask_token_id=199,
        lf_choice_temperature=0.7,
        hf_choice_temperature=0.9,
    )
    token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    monkeypatch.setattr(np.random, "uniform", lambda *_args, **_kwargs: np.array([0.0, 1.0]))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, device=None: torch.tensor(
            [[0.1, 0.9, 0.2, 0.8], [0.7, 0.6, 0.4, 0.5]],
            device=device,
        ),
    )

    masked, known = masking.randomly_mask_tokens(token_ids, mask_token_id=99, device=torch.device("cpu"))

    expected_masked = torch.tensor([[99, 2, 3, 4], [99, 99, 99, 99]])
    expected_known = torch.tensor([[False, True, True, True], [False, False, False, False]])
    assert torch.equal(masked, expected_masked)
    assert torch.equal(known, expected_known)


def test_mask_by_random_topk_selects_lowest_confidence_when_temperature_zero():
    mask_len = torch.tensor([[2.0], [2.0]])
    probs = torch.tensor([[0.9, 0.1, 0.3], [0.2, 0.8, 0.1]])

    masking = maskgit_mod._MaskingLogic(
        lf_mask_token_id=99,
        hf_mask_token_id=199,
        lf_choice_temperature=0.7,
        hf_choice_temperature=0.9,
    )
    topk_mask = masking.mask_by_random_topk(mask_len, probs, temperature=0.0, device="cpu")

    expected = torch.tensor([[False, True, True], [True, False, True]])
    assert torch.equal(topk_mask, expected)


def test_mask_by_random_topk_raises_for_non_homogeneous_mask_len():
    probs = torch.tensor([[0.9, 0.1, 0.3], [0.2, 0.8, 0.1]])
    mask_len = torch.tensor([[1.0], [2.0]])

    masking = maskgit_mod._MaskingLogic(
        lf_mask_token_id=99,
        hf_mask_token_id=199,
        lf_choice_temperature=0.7,
        hf_choice_temperature=0.9,
    )
    with pytest.raises(ValueError, match="homogeneous"):
        masking.mask_by_random_topk(mask_len, probs, temperature=0.0, device="cpu")


class _EvalRecorder:
    def __init__(self, name):
        self.name = name
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1


def test_compute_mask_prediction_loss_masks_and_reduces_cross_entropy():
    token_ids_l = torch.tensor([[0, 1, 2], [2, 1, 0]])
    token_ids_h = torch.tensor([[1, 0, 1], [0, 2, 2]])
    known_l = torch.tensor([[True, False, True], [False, True, False]])
    known_h = torch.tensor([[False, True, False], [True, False, True]])

    token_ids_l_masked = torch.tensor([[0, 10, 2], [10, 1, 10]])
    token_ids_h_masked = torch.tensor([[20, 0, 20], [0, 20, 2]])

    logits_l = torch.tensor(
        [
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 3.0, 1.0]],
        ]
    )
    logits_h = torch.tensor(
        [
            [[0.0, 2.0, 1.0], [2.0, 0.0, 1.0], [1.0, 3.0, 0.0]],
            [[3.0, 0.0, 1.0], [0.0, 1.0, 3.0], [0.0, 2.0, 1.0]],
        ]
    )

    class MaskingStub:
        mask_token_ids = {"lf": 10, "hf": 20}

        def randomly_mask_tokens(self, token_ids, mask_token_id, _device):
            if mask_token_id == 10:
                assert torch.equal(token_ids, token_ids_l)
                return token_ids_l_masked, known_l
            assert mask_token_id == 20
            assert torch.equal(token_ids, token_ids_h)
            return token_ids_h_masked, known_h

    def encode_to_z_q_fn(_x, encoder, _vq):
        return torch.zeros(1), token_ids_l if encoder.name == "lf" else token_ids_h

    def masked_prediction_fn(transformer, _class_condition, *tokens):
        if transformer.name == "lf":
            assert torch.equal(tokens[0], token_ids_l_masked)
            return logits_l
        assert transformer.name == "hf"
        assert torch.equal(tokens[0], token_ids_l)
        assert torch.equal(tokens[1], token_ids_h_masked)
        return logits_h

    training_logic = maskgit_mod._TrainingLogic(
        masking_logic=MaskingStub(),
        encode_to_z_q_fn=encode_to_z_q_fn,
        masked_prediction_fn=masked_prediction_fn,
    )

    x = torch.randn(2, 1, 4)
    class_condition = torch.tensor([[1], [0]])
    encoder_l = _EvalRecorder("lf")
    vq_model_l = _EvalRecorder("lf")
    encoder_h = _EvalRecorder("hf")
    vq_model_h = _EvalRecorder("hf")
    transformer_l = SimpleNamespace(name="lf")
    transformer_h = SimpleNamespace(name="hf")

    losses = training_logic.compute_mask_prediction_loss(
        x=x,
        class_condition=class_condition,
        encoder_l=encoder_l,
        vq_model_l=vq_model_l,
        encoder_h=encoder_h,
        vq_model_h=vq_model_h,
        transformer_l=transformer_l,
        transformer_h=transformer_h,
    )

    expected_l = F.cross_entropy(logits_l[~known_l], token_ids_l[~known_l])
    expected_h = F.cross_entropy(logits_h[~known_h], token_ids_h[~known_h])
    assert torch.isclose(losses.mask_pred_loss_l, expected_l)
    assert torch.isclose(losses.mask_pred_loss_h, expected_h)
    assert torch.isclose(losses.total_mask_prediction_loss, expected_l + expected_h)
    assert encoder_l.eval_calls == 1
    assert vq_model_l.eval_calls == 1
    assert encoder_h.eval_calls == 1
    assert vq_model_h.eval_calls == 1


def test_create_input_tokens_uses_mask_token_id_and_int_dtype():
    masking = maskgit_mod._MaskingLogic(10, 20, 0.7, 0.9)
    sampling = maskgit_mod._SamplingLogic(masking, 1, 1, masked_prediction_fn=lambda *_: None)
    tokens = sampling.create_input_tokens(num_samples=3, num_tokens=4, mask_token_id=10, device="cpu")
    assert tokens.dtype == torch.int64
    assert torch.equal(tokens, torch.full((3, 4), 10, dtype=torch.int64))


def test_first_pass_respects_unknown_tokens_and_remasks(monkeypatch):
    masking = maskgit_mod._MaskingLogic(99, 199, 0.7, 0.9)
    sampling = maskgit_mod._SamplingLogic(masking, lf_num_sampling_steps=1, hf_num_sampling_steps=1, masked_prediction_fn=lambda *_: torch.zeros(1))

    logits_l = torch.tensor([[[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]])
    sampling.masked_prediction_fn = lambda *_args: logits_l

    monkeypatch.setattr(
        torch.distributions.categorical.Categorical,
        "sample",
        lambda self: torch.tensor([[1, 2, 3]], device=self.logits.device),
    )

    call_args = {}

    def fake_mask_by_random_topk(mask_len, probs, temperature=1.0, device="cpu"):
        call_args["mask_len"] = mask_len.clone()
        call_args["probs_shape"] = probs.shape
        call_args["temperature"] = temperature
        call_args["device"] = str(device)
        return torch.tensor([[False, True, False]], device=probs.device)

    masking.mask_by_random_topk = fake_mask_by_random_topk

    token_ids_l = torch.tensor([[99, 1, 99]])
    out = sampling._first_pass(
        token_ids_l=token_ids_l,
        unknown_token_count_l=torch.tensor([2]),
        class_condition=None,
        mask_schedule=lambda _ratio: 0.5,
        transformer_l=SimpleNamespace(),
        device="cpu",
    )

    assert torch.equal(out, torch.tensor([[1, 99, 3]]))
    assert torch.equal(call_args["mask_len"], torch.tensor([[1.0]]))
    assert call_args["probs_shape"] == (1, 3)
    assert call_args["temperature"] == pytest.approx(0.0)
    assert call_args["device"] == "cpu"


def test_second_pass_uses_hf_mask_token(monkeypatch):
    masking = maskgit_mod._MaskingLogic(99, 199, 0.7, 0.9)
    sampling = maskgit_mod._SamplingLogic(masking, lf_num_sampling_steps=1, hf_num_sampling_steps=1, masked_prediction_fn=lambda *_: torch.zeros(1))

    logits_h = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]])
    sampling.masked_prediction_fn = lambda *_args: logits_h

    monkeypatch.setattr(
        torch.distributions.categorical.Categorical,
        "sample",
        lambda self: torch.tensor([[2, 1]], device=self.logits.device),
    )

    def fake_mask_by_random_topk(mask_len, probs, temperature=1.0, device="cpu"):
        assert torch.equal(mask_len, torch.tensor([[1.0]]))
        assert probs.shape == (1, 2)
        assert temperature == pytest.approx(0.0)
        assert str(device) == "cpu"
        return torch.tensor([[True, False]], device=probs.device)

    masking.mask_by_random_topk = fake_mask_by_random_topk

    token_ids_h = torch.tensor([[199, 1]])
    out = sampling._second_pass(
        token_ids_l=torch.tensor([[1, 2]]),
        token_ids_h=token_ids_h,
        unknown_token_count_h=torch.tensor([1]),
        class_condition=None,
        mask_schedule=lambda _ratio: 1.0,
        transformer_h=SimpleNamespace(),
        device="cpu",
    )
    assert torch.equal(out, torch.tensor([[199, 1]]))


def test_sampling_iterative_decoding_builds_class_condition_and_calls_passes(monkeypatch):
    masking = maskgit_mod._MaskingLogic(9, 19, 0.7, 0.9)
    sampling = maskgit_mod._SamplingLogic(masking, lf_num_sampling_steps=2, hf_num_sampling_steps=3, masked_prediction_fn=lambda *_: torch.zeros(1))

    observed = {}

    def fake_first_pass(token_ids_l, unknown_token_count_l, class_condition, _mask_schedule, _transformer_l, _device):
        observed["first_tokens"] = token_ids_l.clone()
        observed["first_unknown"] = unknown_token_count_l.clone()
        observed["class_condition"] = None if class_condition is None else class_condition.clone()
        return torch.full_like(token_ids_l, 3)

    def fake_second_pass(token_ids_l, token_ids_h, unknown_token_count_h, class_condition, _mask_schedule, _transformer_h, _device):
        observed["second_l"] = token_ids_l.clone()
        observed["second_h"] = token_ids_h.clone()
        observed["second_unknown"] = unknown_token_count_h.clone()
        observed["second_class_condition"] = None if class_condition is None else class_condition.clone()
        return torch.full_like(token_ids_h, 7)

    monkeypatch.setattr(sampling, "_first_pass", fake_first_pass)
    monkeypatch.setattr(sampling, "_second_pass", fake_second_pass)

    l_ids, h_ids = sampling.iterative_decoding(
        num_samples=3,
        num_tokens_l=2,
        num_tokens_h=4,
        transformer_l=SimpleNamespace(),
        transformer_h=SimpleNamespace(),
        mode="cosine",
        class_condition=7,
        device="cpu",
    )

    assert torch.equal(observed["first_tokens"], torch.full((3, 2), 9, dtype=torch.int64))
    assert torch.equal(observed["first_unknown"], torch.tensor([2, 2, 2]))
    assert torch.equal(observed["class_condition"], torch.tensor([[7], [7], [7]], dtype=torch.int64))
    assert torch.equal(observed["second_l"], torch.full((3, 2), 3, dtype=torch.int64))
    assert torch.equal(observed["second_h"], torch.full((3, 4), 19, dtype=torch.int64))
    assert torch.equal(observed["second_unknown"], torch.tensor([4, 4, 4]))
    assert torch.equal(observed["second_class_condition"], torch.tensor([[7], [7], [7]], dtype=torch.int64))
    assert torch.equal(l_ids, torch.full((3, 2), 3, dtype=torch.int64))
    assert torch.equal(h_ids, torch.full((3, 4), 7, dtype=torch.int64))


def test_sampling_iterative_decoding_accepts_1d_tensor_condition(monkeypatch):
    masking = maskgit_mod._MaskingLogic(9, 19, 0.7, 0.9)
    sampling = maskgit_mod._SamplingLogic(masking, lf_num_sampling_steps=1, hf_num_sampling_steps=1, masked_prediction_fn=lambda *_: torch.zeros(1))

    observed = {}

    def fake_first_pass(_token_ids_l, _unknown_token_count_l, class_condition, _mask_schedule, _transformer_l, _device):
        observed["first_class_condition"] = class_condition.clone()
        return torch.zeros((2, 3), dtype=torch.int64)

    def fake_second_pass(_token_ids_l, token_ids_h, _unknown_token_count_h, class_condition, _mask_schedule, _transformer_h, _device):
        observed["second_class_condition"] = class_condition.clone()
        return token_ids_h

    monkeypatch.setattr(sampling, "_first_pass", fake_first_pass)
    monkeypatch.setattr(sampling, "_second_pass", fake_second_pass)

    sampling.iterative_decoding(
        num_samples=2,
        num_tokens_l=3,
        num_tokens_h=4,
        transformer_l=SimpleNamespace(),
        transformer_h=SimpleNamespace(),
        class_condition=torch.tensor([5, 6]),
        device="cpu",
    )

    expected = torch.tensor([[5], [6]], dtype=torch.int64)
    assert torch.equal(observed["first_class_condition"], expected)
    assert torch.equal(observed["second_class_condition"], expected)


def test_normalize_class_condition_int_and_tensor_shapes():
    normalize = maskgit_mod._SamplingLogic._normalize_class_condition

    cond_int = normalize(4, num_samples=3, device="cpu")
    assert torch.equal(cond_int, torch.tensor([[4], [4], [4]], dtype=torch.int64))

    cond_scalar_tensor = normalize(torch.tensor(2), num_samples=3, device="cpu")
    assert torch.equal(cond_scalar_tensor, torch.tensor([[2], [2], [2]], dtype=torch.int64))

    cond_1d_single = normalize(torch.tensor([7]), num_samples=3, device="cpu")
    assert torch.equal(cond_1d_single, torch.tensor([[7], [7], [7]], dtype=torch.int64))

    cond_2d_single = normalize(torch.tensor([[8]]), num_samples=3, device="cpu")
    assert torch.equal(cond_2d_single, torch.tensor([[8], [8], [8]], dtype=torch.int64))


def test_normalize_class_condition_tensor_batch_shapes():
    normalize = maskgit_mod._SamplingLogic._normalize_class_condition
    cond_1d_batch = normalize(torch.tensor([1, 2, 3]), num_samples=3, device="cpu")
    cond_2d_batch = normalize(torch.tensor([[1], [2], [3]]), num_samples=3, device="cpu")
    expected = torch.tensor([[1], [2], [3]], dtype=torch.int64)
    assert torch.equal(cond_1d_batch, expected)
    assert torch.equal(cond_2d_batch, expected)


def test_normalize_class_condition_invalid_inputs_raise():
    normalize = maskgit_mod._SamplingLogic._normalize_class_condition

    with pytest.raises(TypeError):
        normalize("bad", num_samples=2, device="cpu")

    with pytest.raises(ValueError):
        normalize(torch.tensor([1, 2, 3]), num_samples=2, device="cpu")

    with pytest.raises(ValueError):
        normalize(torch.tensor([[1, 2]]), num_samples=2, device="cpu")

    with pytest.raises(ValueError):
        normalize(torch.ones((1, 1, 1)), num_samples=2, device="cpu")


class _FakeBT:
    inits = []

    def __init__(self, *args, **kwargs):
        _FakeBT.inits.append((args, kwargs))

    def __call__(self, *token_inputs, class_condition=None):
        b, n = token_inputs[0].shape
        return torch.zeros((b, n, 4))


class _TinyEncoder(torch.nn.Module):
    def __init__(self, num_tokens, h_prime, w_prime, offset):
        super().__init__()
        self.num_tokens = torch.tensor(num_tokens)
        self.H_prime = torch.tensor(h_prime)
        self.W_prime = torch.tensor(w_prime)
        self.offset = offset

    def forward(self, x):
        return x + self.offset


class _TinyDecoder(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, zq):
        return zq + self.offset


class _TinyVQ(torch.nn.Module):
    def __init__(self, embed, scale):
        super().__init__()
        self._codebook = SimpleNamespace(embed=embed)
        self.scale = scale

    def project_out(self, zq):
        return zq * self.scale


class _TinyVQVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_l = _TinyEncoder(num_tokens=4, h_prime=2, w_prime=2, offset=1.0)
        self.decoder_l = _TinyDecoder(offset=10.0)
        self.vq_model_l = _TinyVQ(
            embed=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                    [3.0, 1.0],
                ]
            ),
            scale=2.0,
        )
        self.encoder_h = _TinyEncoder(num_tokens=6, h_prime=2, w_prime=3, offset=2.0)
        self.decoder_h = _TinyDecoder(offset=20.0)
        self.vq_model_h = _TinyVQ(
            embed=torch.tensor(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 2.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [4.0, 4.0],
                    [5.0, 5.0],
                ]
            ),
            scale=3.0,
        )
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


@pytest.fixture
def maskgit_instance(monkeypatch):
    _FakeBT.inits.clear()
    freeze_calls = []
    monkeypatch.setattr(maskgit_mod, "BidirectionalTransformer", _FakeBT)
    monkeypatch.setattr(maskgit_mod, "freeze", lambda model: freeze_calls.append(model))

    vqvae = _TinyVQVAE()
    lf_prior_cfg = PriorModelConfig(
        hidden_dim=32,
        n_layers=2,
        heads=2,
        ff_mult=2,
        use_rmsnorm=False,
        p_unconditional=0.1,
    )
    hf_prior_cfg = PriorModelConfig(
        hidden_dim=64,
        n_layers=2,
        heads=2,
        ff_mult=2,
        use_rmsnorm=False,
        p_unconditional=0.1,
    )
    model = maskgit_mod.MaskGIT(
        vqvae=vqvae,
        lf_choice_temperature=0.7,
        hf_choice_temperature=0.9,
        lf_num_sampling_steps=2,
        hf_num_sampling_steps=3,
        lf_codebook_size=5,
        hf_codebook_size=7,
        transformer_embedding_dim=16,
        lf_prior_model_config=lf_prior_cfg,
        hf_prior_model_config=hf_prior_cfg,
        classifier_free_guidance_scale=2.5,
        n_classes=4,
    )
    return model, vqvae, freeze_calls


def test_maskgit_init_freezes_vqvae_and_builds_transformers(maskgit_instance):
    model, vqvae, freeze_calls = maskgit_instance
    assert freeze_calls == [vqvae]
    assert vqvae.eval_calls == 1
    assert model.num_tokens_l == 4
    assert model.num_tokens_h == 6
    assert len(_FakeBT.inits) == 2

    lf_args, lf_kwargs = _FakeBT.inits[0]
    hf_args, hf_kwargs = _FakeBT.inits[1]
    assert lf_args[0] == "lf"
    assert hf_args[0] == "hf"
    assert lf_kwargs["n_classes"] == 4
    assert hf_kwargs["n_classes"] == 4
    assert hf_kwargs["num_tokens_l"] == 4


def test_encode_to_z_q_uses_quantize(monkeypatch, maskgit_instance):
    model, _, _ = maskgit_instance

    observed = {}

    def fake_quantize(z, vq_model, svq_temp=None):
        observed["z"] = z.clone()
        observed["vq_model"] = vq_model
        observed["svq_temp"] = svq_temp
        return z + 10.0, torch.tensor([[1, 2, 3]]), None, None

    monkeypatch.setattr(maskgit_mod, "quantize", fake_quantize)
    x = torch.zeros(1, 2, 3)
    encoder = _TinyEncoder(num_tokens=3, h_prime=1, w_prime=3, offset=5.0)
    vq_model = object()

    zq, token_ids = model.encode_to_z_q(x, encoder=encoder, vq_model=vq_model, svq_temp=0.3)
    assert torch.equal(observed["z"], x + 5.0)
    assert observed["vq_model"] is vq_model
    assert observed["svq_temp"] == 0.3
    assert torch.equal(zq, x + 15.0)
    assert torch.equal(token_ids, torch.tensor([[1, 2, 3]]))


def test_masked_prediction_none_condition_is_unconditional(maskgit_instance):
    model, _, _ = maskgit_instance

    class FakeTransformer:
        def __init__(self):
            self.calls = []

        def __call__(self, *token_inputs, class_condition=None):
            self.calls.append(class_condition)
            b, n = token_inputs[0].shape
            return torch.full((b, n, 2), 3.0)

    transformer = FakeTransformer()
    logits = model.masked_prediction(transformer, None, torch.tensor([[1, 2]]))
    assert torch.equal(logits, torch.full((1, 2, 2), 3.0))
    assert transformer.calls == [None]


def test_masked_prediction_cfg_scale_one_calls_conditional_once(maskgit_instance):
    model, _, _ = maskgit_instance
    model.cfg_scale = 1.0

    class FakeTransformer:
        def __init__(self):
            self.calls = []

        def __call__(self, *token_inputs, class_condition=None):
            self.calls.append(class_condition)
            b, n = token_inputs[0].shape
            return torch.full((b, n, 2), 5.0 if class_condition is not None else -1.0)

    transformer = FakeTransformer()
    cond = torch.tensor([[2]])
    logits = model.masked_prediction(transformer, cond, torch.tensor([[1, 2]]))
    assert torch.equal(logits, torch.full((1, 2, 2), 5.0))
    assert len(transformer.calls) == 1
    assert torch.equal(transformer.calls[0], cond)


def test_masked_prediction_cfg_guidance_mix(maskgit_instance):
    model, _, _ = maskgit_instance
    model.cfg_scale = 3.0

    class FakeTransformer:
        def __init__(self):
            self.calls = []

        def __call__(self, *token_inputs, class_condition=None):
            self.calls.append(class_condition)
            b, n = token_inputs[0].shape
            if class_condition is None:
                return torch.full((b, n, 2), 1.0)
            return torch.full((b, n, 2), 3.0)

    transformer = FakeTransformer()
    cond = torch.tensor([[1]])
    logits = model.masked_prediction(transformer, cond, torch.tensor([[2, 3]]))
    assert torch.equal(logits, torch.full((1, 2, 2), 7.0))
    assert len(transformer.calls) == 2


def test_maskgit_forward_delegates_to_training_logic(maskgit_instance, monkeypatch):
    model, _, _ = maskgit_instance
    expected = maskgit_mod.MaskPredictionLoss(
        total_mask_prediction_loss=torch.tensor(9.0),
        mask_pred_loss_l=torch.tensor(4.0),
        mask_pred_loss_h=torch.tensor(5.0),
    )
    captured = {}

    def fake_compute(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(model.training_logic, "compute_mask_prediction_loss", fake_compute)
    x = torch.randn(2, 1, 8)
    class_condition = torch.tensor([[1], [0]])
    out = model.forward(x, class_condition)
    assert out == expected
    assert captured["x"] is x
    assert captured["class_condition"] is class_condition
    assert captured["encoder_l"] is model.encoder_l
    assert captured["transformer_h"] is model.transformer_h


def test_maskgit_iterative_decoding_delegates_to_sampling_logic(maskgit_instance, monkeypatch):
    model, _, _ = maskgit_instance
    expected = (torch.tensor([[1, 2]]), torch.tensor([[3, 4, 5]]))
    captured = {}

    def fake_iterative_decoding(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(model.sampling_logic, "iterative_decoding", fake_iterative_decoding)
    out = model.iterative_decoding(num_samples=5, mode="square", class_condition=3, device="cpu")
    assert out == expected
    assert captured["num_samples"] == 5
    assert captured["num_tokens_l"] == model.num_tokens_l
    assert captured["num_tokens_h"] == model.num_tokens_h
    assert captured["mode"] == "square"
    assert captured["class_condition"] == 3


def test_decode_token_ind_to_timeseries_lf_and_return_representations(maskgit_instance):
    model, _, _ = maskgit_instance
    token_ids = torch.tensor([[1, 2, 3, 4]])

    xhat, zq = model.decode_token_ind_to_timeseries(token_ids, frequency="lf", return_representations=True)

    expected_embed = F.embedding(token_ids, model.vq_model_l._codebook.embed)
    expected_embed = model.vq_model_l.project_out(expected_embed)
    expected_zq = torch.einsum("bnc->bcn", expected_embed).reshape(1, 2, 2, 2)
    assert torch.equal(zq, expected_zq)
    assert torch.equal(xhat, expected_zq + 10.0)


def test_decode_token_ind_to_timeseries_hf_path(maskgit_instance):
    model, _, _ = maskgit_instance
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    xhat = model.decode_token_ind_to_timeseries(token_ids, frequency="hf")
    assert xhat.shape == (1, 2, 2, 3)
    assert torch.all(xhat >= 20.0)


def test_decode_token_ind_to_timeseries_invalid_frequency_raises(maskgit_instance):
    model, _, _ = maskgit_instance
    with pytest.raises(AssertionError):
        model.decode_token_ind_to_timeseries(torch.tensor([[1, 2, 3, 4]]), frequency="mid")
