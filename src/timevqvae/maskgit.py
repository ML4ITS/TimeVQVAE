from dataclasses import asdict, dataclass
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timevqvae.models.bidirectional_transformer import BidirectionalTransformer
from timevqvae.models.vq_vae_encdec import VQVAEEncoder
from timevqvae.utils.nn import freeze, quantize
from timevqvae.vector_quantization.vq import VectorQuantize
from timevqvae.vqvae import VQVAE


@dataclass(frozen=True)
class PriorModelConfig:
    """Schema for LF/HF prior transformer hyperparameters."""

    hidden_dim: int
    n_layers: int
    heads: int
    ff_mult: int
    use_rmsnorm: bool
    p_unconditional: float
    model_dropout: float = 0.3
    emb_dropout: float = 0.3


@dataclass(frozen=True)
class MaskPredictionLoss:
    """Schema for mask prediction losses."""

    total_mask_prediction_loss: torch.Tensor
    mask_pred_loss_l: torch.Tensor
    mask_pred_loss_h: torch.Tensor


class _MaskingLogic:
    """Token masking utilities used by both training and iterative decoding."""

    def __init__(
        self,
        lf_mask_token_id: int,
        hf_mask_token_id: int,
        lf_choice_temperature: float,
        hf_choice_temperature: float,
    ):
        self.mask_token_ids = {
            "lf": lf_mask_token_id,
            "hf": hf_mask_token_id,
        }
        self.choice_temperatures = {
            "lf": lf_choice_temperature,
            "hf": hf_choice_temperature,
        }
        self.training_mask_ratio_schedule = self.gamma_func("cosine")

    @staticmethod
    def gamma_func(mode: str = "cosine") -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
        if mode == "linear":
            return lambda r: 1 - r
        if mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        if mode == "square":
            return lambda r: 1 - r**2
        if mode == "cubic":
            return lambda r: 1 - r**3
        raise NotImplementedError

    def randomly_mask_tokens(self, token_ids: torch.Tensor, mask_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        token_ids: (b n)
        returns:
            masked_token_ids: (b n)
            known_token_mask: (b n) True for known/unmasked
        """
        batch_size, num_tokens = token_ids.shape

        # Sample how many tokens to keep and then mask the rest.
        ratios = np.random.uniform(0, 1, (batch_size,))
        n_known_tokens = np.floor(self.training_mask_ratio_schedule(ratios) * num_tokens)
        n_known_tokens = np.clip(n_known_tokens, a_min=0, a_max=num_tokens - 1).astype(int)

        random_scores = torch.rand((batch_size, num_tokens), device=device)
        known_token_mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device)
        for i in range(batch_size):
            keep_indices = random_scores[i].topk(n_known_tokens[i], dim=-1).indices
            known_token_mask[i].scatter_(dim=-1, index=keep_indices, value=True)

        mask_fill = mask_token_id * torch.ones((batch_size, num_tokens), device=device)
        masked_token_ids = known_token_mask * token_ids + (~known_token_mask) * mask_fill
        return masked_token_ids.long(), known_token_mask

    @staticmethod
    def mask_by_random_topk(
        mask_len: torch.Tensor,
        probs: torch.Tensor,
        temperature: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        mask_len: (b 1)
        probs: (b n)

        `mask_len` is expected to be homogeneous across batch (same value for all items).
        """

        def log(t: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t: torch.Tensor) -> torch.Tensor:
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        if mask_len.ndim != 2 or mask_len.shape[1] != 1:
            raise ValueError("`mask_len` must have shape (batch_size, 1).")
        if mask_len.shape[0] != probs.shape[0]:
            raise ValueError("`mask_len` batch size must match `probs` batch size.")
        if not torch.all(mask_len == mask_len[0]):
            raise ValueError("`mask_len` must be homogeneous across the batch.")

        confidence = torch.log(probs + 1e-5) + temperature * gumbel_noise(probs).to(device)
        mask_len_scalar = int(mask_len[0, 0].item())
        if mask_len_scalar < 0 or mask_len_scalar > probs.shape[-1]:
            raise ValueError("`mask_len` must be in [0, num_tokens].")

        masking_indices = torch.topk(confidence, k=mask_len_scalar, dim=-1, largest=False).indices
        masking = torch.zeros_like(confidence).to(device)
        for i in range(masking_indices.shape[0]):
            masking[i, masking_indices[i].long()] = 1.0
        return masking.bool()


class _TrainingLogic:
    """Training-only masked prediction loss computation."""

    def __init__(
        self,
        masking_logic: _MaskingLogic,
        encode_to_z_q_fn: Callable,
        masked_prediction_fn: Callable,
    ):
        self.masking_logic = masking_logic
        self.encode_to_z_q_fn = encode_to_z_q_fn
        self.masked_prediction_fn = masked_prediction_fn

    def compute_mask_prediction_loss(
        self,
        x: torch.Tensor,
        class_condition: torch.Tensor,
        encoder_l: VQVAEEncoder,
        vq_model_l: VectorQuantize,
        encoder_h: VQVAEEncoder,
        vq_model_h: VectorQuantize,
        transformer_l: BidirectionalTransformer,
        transformer_h: BidirectionalTransformer,
    ) -> MaskPredictionLoss:
        encoder_l.eval()
        vq_model_l.eval()
        encoder_h.eval()
        vq_model_h.eval()

        device = x.device
        _, token_ids_l = self.encode_to_z_q_fn(x, encoder_l, vq_model_l)
        _, token_ids_h = self.encode_to_z_q_fn(x, encoder_h, vq_model_h)

        token_ids_l_masked, known_mask_l = self.masking_logic.randomly_mask_tokens(
            token_ids_l,
            self.masking_logic.mask_token_ids["lf"],
            device,
        )
        token_ids_h_masked, known_mask_h = self.masking_logic.randomly_mask_tokens(
            token_ids_h,
            self.masking_logic.mask_token_ids["hf"],
            device,
        )

        logits_l = self.masked_prediction_fn(transformer_l, class_condition, token_ids_l_masked)
        logits_h = self.masked_prediction_fn(transformer_h, class_condition, token_ids_l, token_ids_h_masked)

        logits_l_on_masked = logits_l[~known_mask_l]
        token_ids_l_on_masked = token_ids_l[~known_mask_l]
        mask_pred_loss_l = F.cross_entropy(logits_l_on_masked.float(), token_ids_l_on_masked.long())

        logits_h_on_masked = logits_h[~known_mask_h]
        token_ids_h_on_masked = token_ids_h[~known_mask_h]
        mask_pred_loss_h = F.cross_entropy(logits_h_on_masked.float(), token_ids_h_on_masked.long())

        total_mask_prediction_loss = mask_pred_loss_l + mask_pred_loss_h
        return MaskPredictionLoss(
            total_mask_prediction_loss=total_mask_prediction_loss,
            mask_pred_loss_l=mask_pred_loss_l,
            mask_pred_loss_h=mask_pred_loss_h,
        )


class _SamplingLogic:
    """Iterative decoding logic used for inference/sampling."""

    def __init__(
        self,
        masking_logic: _MaskingLogic,
        lf_num_sampling_steps: int,
        hf_num_sampling_steps: int,
        masked_prediction_fn: Callable,
    ):
        self.masking_logic = masking_logic
        self.num_sampling_steps = {
            "lf": lf_num_sampling_steps,
            "hf": hf_num_sampling_steps,
        }
        self.masked_prediction_fn = masked_prediction_fn

    @staticmethod
    def create_input_tokens(num_samples: int, num_tokens: int, mask_token_id: int, device: Union[str, torch.device]) -> torch.Tensor:
        blank_tokens = torch.ones((num_samples, num_tokens), device=device)
        masked_tokens = mask_token_id * blank_tokens
        return masked_tokens.to(torch.int64)

    def _first_pass(
        self,
        token_ids_l: torch.Tensor,
        unknown_token_count_l: torch.Tensor,
        class_condition: Union[torch.Tensor, None],
        mask_schedule: Callable,
        transformer_l: BidirectionalTransformer,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        for step in range(self.num_sampling_steps["lf"]):
            logits_l = self.masked_prediction_fn(transformer_l, class_condition, token_ids_l)

            sampled_ids = torch.distributions.categorical.Categorical(logits=logits_l).sample()
            unknown_map = token_ids_l == self.masking_logic.mask_token_ids["lf"]
            sampled_ids = torch.where(unknown_map, sampled_ids, token_ids_l)

            step_ratio = 1.0 * (step + 1) / self.num_sampling_steps["lf"]
            mask_ratio = mask_schedule(step_ratio)

            probs = F.softmax(logits_l, dim=-1)
            selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze()
            confidence_of_known_tokens = torch.tensor([torch.inf], device=device)
            selected_probs = torch.where(unknown_map, selected_probs, confidence_of_known_tokens)

            mask_len = torch.unsqueeze(torch.floor(unknown_token_count_l * mask_ratio), 1)
            mask_len = torch.clip(mask_len, min=0.0)

            remask = self.masking_logic.mask_by_random_topk(
                mask_len,
                selected_probs,
                temperature=self.masking_logic.choice_temperatures["lf"] * (1.0 - step_ratio),
                device=device,
            )
            token_ids_l = torch.where(remask, self.masking_logic.mask_token_ids["lf"], sampled_ids)

        return token_ids_l

    def _second_pass(
        self,
        token_ids_l: torch.Tensor,
        token_ids_h: torch.Tensor,
        unknown_token_count_h: torch.Tensor,
        class_condition: Union[torch.Tensor, None],
        mask_schedule: Callable,
        transformer_h: BidirectionalTransformer,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        for step in range(self.num_sampling_steps["hf"]):
            logits_h = self.masked_prediction_fn(transformer_h, class_condition, token_ids_l, token_ids_h)

            sampled_ids = torch.distributions.categorical.Categorical(logits=logits_h).sample()
            unknown_map = token_ids_h == self.masking_logic.mask_token_ids["hf"]
            sampled_ids = torch.where(unknown_map, sampled_ids, token_ids_h)

            step_ratio = 1.0 * (step + 1) / self.num_sampling_steps["hf"]
            mask_ratio = mask_schedule(step_ratio)

            probs = F.softmax(logits_h, dim=-1)
            selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze()
            confidence_of_known_tokens = torch.tensor([torch.inf], device=device)
            selected_probs = torch.where(unknown_map, selected_probs, confidence_of_known_tokens)

            mask_len = torch.unsqueeze(torch.floor(unknown_token_count_h * mask_ratio), 1)
            mask_len = torch.clip(mask_len, min=0.0)

            remask = self.masking_logic.mask_by_random_topk(
                mask_len,
                selected_probs,
                temperature=self.masking_logic.choice_temperatures["hf"] * (1.0 - step_ratio),
                device=device,
            )
            token_ids_h = torch.where(remask, self.masking_logic.mask_token_ids["hf"], sampled_ids)

        return token_ids_h

    @torch.no_grad()
    def iterative_decoding(
        self,
        num_samples: int,
        num_tokens_l: int,
        num_tokens_h: int,
        transformer_l: BidirectionalTransformer,
        transformer_h: BidirectionalTransformer,
        mode: str = "cosine",
        class_condition: Union[int, torch.Tensor, None] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids_l = self.create_input_tokens(num_samples, num_tokens_l, self.masking_logic.mask_token_ids["lf"], device)
        token_ids_h = self.create_input_tokens(num_samples, num_tokens_h, self.masking_logic.mask_token_ids["hf"], device)

        unknown_token_count_l = torch.sum(token_ids_l == self.masking_logic.mask_token_ids["lf"], dim=-1)
        unknown_token_count_h = torch.sum(token_ids_h == self.masking_logic.mask_token_ids["hf"], dim=-1)
        mask_schedule = self.masking_logic.gamma_func(mode)

        class_condition = self._normalize_class_condition(class_condition, num_samples, device)

        token_ids_l = self._first_pass(
            token_ids_l,
            unknown_token_count_l,
            class_condition,
            mask_schedule,
            transformer_l,
            device,
        )
        token_ids_h = self._second_pass(
            token_ids_l,
            token_ids_h,
            unknown_token_count_h,
            class_condition,
            mask_schedule,
            transformer_h,
            device,
        )
        return token_ids_l, token_ids_h

    @staticmethod
    def _normalize_class_condition(
        class_condition: Union[int, torch.Tensor, None],
        num_samples: int,
        device: Union[str, torch.device],
    ) -> Union[torch.Tensor, None]:
        if class_condition is None:
            return None

        if isinstance(class_condition, int):
            return torch.full((num_samples, 1), class_condition, device=device, dtype=torch.long)

        if not torch.is_tensor(class_condition):
            raise TypeError("`class_condition` must be None, int, or torch.Tensor.")

        class_condition = class_condition.to(device=device).long()
        if class_condition.ndim == 0:
            return class_condition.view(1, 1).repeat(num_samples, 1)
        if class_condition.ndim == 1:
            if class_condition.shape[0] == 1:
                return class_condition.view(1, 1).repeat(num_samples, 1)
            if class_condition.shape[0] == num_samples:
                return class_condition.unsqueeze(1)
            raise ValueError("1D `class_condition` must have length 1 or `num_samples`.")
        if class_condition.ndim == 2:
            if class_condition.shape[1] != 1:
                raise ValueError("2D `class_condition` must have shape (batch_size, 1).")
            if class_condition.shape[0] == 1:
                return class_condition.repeat(num_samples, 1)
            if class_condition.shape[0] == num_samples:
                return class_condition
            raise ValueError("2D `class_condition` batch size must be 1 or `num_samples`.")

        raise ValueError("`class_condition` must be a scalar, 1D, or 2D tensor.")


class MaskGIT(nn.Module):
    """Orchestrator for MaskGIT training and sampling."""

    def __init__(
        self,
        vqvae: VQVAE,
        lf_choice_temperature: float,
        hf_choice_temperature: float,
        lf_num_sampling_steps: int,
        hf_num_sampling_steps: int,
        lf_codebook_size: int,
        hf_codebook_size: int,
        transformer_embedding_dim: int,
        lf_prior_model_config: PriorModelConfig,
        hf_prior_model_config: PriorModelConfig,
        classifier_free_guidance_scale: float,
        n_classes: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.cfg_scale = classifier_free_guidance_scale

        self.vqvae = vqvae
        freeze(self.vqvae)
        self.vqvae.eval()

        self.encoder_l = self.vqvae.encoder_l
        self.decoder_l = self.vqvae.decoder_l
        self.vq_model_l = self.vqvae.vq_model_l
        self.encoder_h = self.vqvae.encoder_h
        self.decoder_h = self.vqvae.decoder_h
        self.vq_model_h = self.vqvae.vq_model_h

        self.num_tokens_l = self.encoder_l.num_tokens.item()
        self.num_tokens_h = self.encoder_h.num_tokens.item()

        self.H_prime_l = self.encoder_l.H_prime.item()
        self.H_prime_h = self.encoder_h.H_prime.item()
        self.W_prime_l = self.encoder_l.W_prime.item()
        self.W_prime_h = self.encoder_h.W_prime.item()

        self.masking_logic = _MaskingLogic(
            lf_mask_token_id=lf_codebook_size,
            hf_mask_token_id=hf_codebook_size,
            lf_choice_temperature=lf_choice_temperature,
            hf_choice_temperature=hf_choice_temperature,
        )

        codebook_sizes = {
            "lf": lf_codebook_size,
            "hf": hf_codebook_size,
        }
        lf_prior_model_config_dict = self._normalize_prior_model_config(lf_prior_model_config, "lf_prior_model_config")
        hf_prior_model_config_dict = self._normalize_prior_model_config(hf_prior_model_config, "hf_prior_model_config")

        self.transformer_l = BidirectionalTransformer(
            "lf",
            self.num_tokens_l,
            codebook_sizes,
            transformer_embedding_dim,
            **lf_prior_model_config_dict,
            n_classes=n_classes,
        )
        self.transformer_h = BidirectionalTransformer(
            "hf",
            self.num_tokens_h,
            codebook_sizes,
            transformer_embedding_dim,
            **hf_prior_model_config_dict,
            n_classes=n_classes,
            num_tokens_l=self.num_tokens_l,
        )

        self.training_logic = _TrainingLogic(
            masking_logic=self.masking_logic,
            encode_to_z_q_fn=self.encode_to_z_q,
            masked_prediction_fn=self.masked_prediction,
        )
        self.sampling_logic = _SamplingLogic(
            masking_logic=self.masking_logic,
            lf_num_sampling_steps=lf_num_sampling_steps,
            hf_num_sampling_steps=hf_num_sampling_steps,
            masked_prediction_fn=self.masked_prediction,
        )

    @staticmethod
    def _normalize_prior_model_config(
        config: PriorModelConfig,
        config_name: str,
    ) -> dict:
        if isinstance(config, PriorModelConfig):
            return asdict(config)
        raise TypeError(f"{config_name} must be a PriorModelConfig instance.")

    @torch.no_grad()
    def encode_to_z_q(
        self,
        x: torch.Tensor,
        encoder: VQVAEEncoder,
        vq_model: VectorQuantize,
        svq_temp: Union[float, None] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = encoder(x)
        zq, token_ids, _, _ = quantize(z, vq_model, svq_temp=svq_temp)
        return zq, token_ids

    def masked_prediction(self, transformer: BidirectionalTransformer, class_condition: Union[torch.Tensor, None], *token_inputs: torch.Tensor) -> torch.Tensor:
        if class_condition is None:
            return transformer(*token_inputs, class_condition=None)

        if self.cfg_scale == 1.0:
            return transformer(*token_inputs, class_condition=class_condition)

        logits_unconditional = transformer(*token_inputs, class_condition=None)
        logits_conditional = transformer(*token_inputs, class_condition=class_condition)
        return logits_unconditional + self.cfg_scale * (logits_conditional - logits_unconditional)

    def forward(self, x: torch.Tensor, class_condition: torch.Tensor) -> MaskPredictionLoss:
        """
        Compute the masked token prediction losses for low- and high-frequency priors.

        Args:
            x: Input time-series batch.
            class_condition: Class-conditioning tensor.

        Returns:
            MaskPredictionLoss with:
                - total_mask_prediction_loss: Sum of LF and HF cross-entropy losses.
                - mask_pred_loss_l: LF cross-entropy loss.
                - mask_pred_loss_h: HF cross-entropy loss.
        """
        return self.training_logic.compute_mask_prediction_loss(
            x=x,
            class_condition=class_condition,
            encoder_l=self.encoder_l,
            vq_model_l=self.vq_model_l,
            encoder_h=self.encoder_h,
            vq_model_h=self.vq_model_h,
            transformer_l=self.transformer_l,
            transformer_h=self.transformer_h,
        )

    @torch.no_grad()
    def iterative_decoding(
        self,
        num_samples: int = 1,
        mode: str = "cosine",
        class_condition: Union[int, torch.Tensor, None] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sampling_logic.iterative_decoding(
            num_samples=num_samples,
            num_tokens_l=self.num_tokens_l,
            num_tokens_h=self.num_tokens_h,
            transformer_l=self.transformer_l,
            transformer_h=self.transformer_h,
            mode=mode,
            class_condition=class_condition,
            device=device,
        )

    def decode_token_ind_to_timeseries(
        self,
        token_ids: torch.Tensor,
        frequency: str,
        return_representations: bool = False,
    ):
        self.eval()
        frequency = frequency.lower()
        assert frequency in ["lf", "hf"]

        vq_model = self.vq_model_l if frequency == "lf" else self.vq_model_h
        decoder = self.decoder_l if frequency == "lf" else self.decoder_h

        zq = F.embedding(token_ids, vq_model._codebook.embed)
        zq = vq_model.project_out(zq)
        zq = rearrange(zq, "b n c -> b c n")

        h_prime = self.H_prime_l if frequency == "lf" else self.H_prime_h
        w_prime = self.W_prime_l if frequency == "lf" else self.W_prime_h
        zq = rearrange(zq, "b c (h w) -> b c h w", h=h_prime, w=w_prime)

        xhat = decoder(zq)
        if return_representations:
            return xhat, zq
        return xhat
