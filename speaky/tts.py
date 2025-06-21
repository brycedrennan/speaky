import torch
import torch.nn.functional as F
from chatterbox.tts import ChatterboxTTS, T3Cond, drop_invalid_tokens, punc_norm


def auto_batch_size(device: str) -> int:
    """Return a best-guess batch size for ``device``."""
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        gb = props.total_memory / (1024**3)
        if gb >= 20:
            return 8
        if gb >= 10:
            return 4
        return 2
    return 1


class ParallelChatterboxTTS(ChatterboxTTS):
    """Extend :class:`ChatterboxTTS` with a vectorised `generate_batch`."""

    # ---------- helpers -------------------------------------------------
    def _prepare_text_batch(self, texts: list[str], cfg_weight: float) -> torch.Tensor:
        """
        Normalise, tokenize and pad a list of texts, returning
        a tensor of shape (B  *or* 2 B, L+2) on the correct device.
        (the +2 accounts for <s> and </s> tokens)
        """
        # 1. punctuation / spacing cleanup
        texts = [punc_norm(t) for t in texts]

        # 2. convert each string → tensor([tokens])
        tok_seqs = [self.tokenizer.text_to_tokens(t) for t in texts]

        # 3. pad to common length
        max_len = max(t.size(-1) for t in tok_seqs)
        pad_val = self.t3.hp.pad_text_token
        tok_seqs = [F.pad(t, (0, max_len - t.size(-1)), value=pad_val) for t in tok_seqs]
        tokens = torch.stack(tok_seqs, 0).to(self.device)  # (B, L)

        # 4. classifier-free guidance duplication (optional)
        if cfg_weight > 0:
            tokens = torch.cat([tokens, tokens], 0)  # (2 B, L)

        # 5. add <s> .. </s>
        sot, eot = self.t3.hp.start_text_token, self.t3.hp.stop_text_token
        tokens = F.pad(tokens, (1, 0), value=sot)
        tokens = F.pad(tokens, (0, 1), value=eot)  # (B | 2 B, L+2)

        return tokens

    # ---------- public API ----------------------------------------------
    @torch.inference_mode()
    def generate_batch(
        self,
        texts: list[str],
        *,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        audio_prompt_paths: list[str] | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> list[torch.Tensor]:
        """
        Vectorised version of `generate`.

        Parameters
        ----------
        texts
            list of N text chunks to synthesise.
        audio_prompt_paths
            optional list of N reference-audio paths.  If omitted the
            instance`s existing conditionals are reused for every item.
        Returns
        -------
        list[torch.Tensor]
            length-N list; each tensor has shape (1, num_samples).
        """
        B = len(texts)

        # ----- build / replicate conditionals ---------------------------
        if audio_prompt_paths is not None:
            assert len(audio_prompt_paths) == B, "`audio_prompt_paths` must have the same length as `texts`."
            conds = []
            for path in audio_prompt_paths:
                self.prepare_conditionals(path, exaggeration=exaggeration)
                # `prepare_conditionals` sets self.conds - copy so that we
                # can stack after the loop
                conds.append(self.conds)
        else:
            if self.conds is None:
                msg = "Must call `prepare_conditionals()` or pass `audio_prompt_paths`."
                raise RuntimeError(msg)
            conds = [self.conds] * B

        # ----- stack T3 conditionals ------------------------------------
        t3_cond = T3Cond(
            speaker_emb=torch.cat([c.t3.speaker_emb for c in conds], 0),
            cond_prompt_speech_tokens=(
                torch.cat([c.t3.cond_prompt_speech_tokens for c in conds], 0)
                if conds[0].t3.cond_prompt_speech_tokens is not None
                else None
            ),
            emotion_adv=exaggeration * torch.ones(B, 1, 1, device=self.device),
        )

        # ----- stack S3Gen reference dict -------------------------------
        ref_dict = {
            k: (torch.cat([c.gen[k] for c in conds], 0) if torch.is_tensor(conds[0].gen[k]) else conds[0].gen[k])
            for k in conds[0].gen
        }

        # ----- text tokens ----------------------------------------------
        text_tokens = self._prepare_text_batch(texts, cfg_weight)

        # ----- T3 inference ---------------------------------------------
        speech_tok_batch = self.t3.inference(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )

        # discard the duplicated half if CFG was enabled
        speech_tok_batch = speech_tok_batch[:B]

        # clean + clip illegal tokens
        cleaned = []
        for seq in speech_tok_batch:
            seq = drop_invalid_tokens(seq)
            cleaned.append(seq[seq < 6561].to(self.device))

        # Pad to a uniform length for the generator
        padded = torch.nn.utils.rnn.pad_sequence(cleaned, batch_first=True)

        # ----- S3Gen inference ------------------------------------------
        wavs, _ = self.s3gen.inference(
            speech_tokens=padded,
            ref_dict=ref_dict,
        )  # (B, T)

        wavs = wavs.cpu().numpy()

        # ----- watermark & return ---------------------------------------
        result = [
            torch.from_numpy(self.watermarker.apply_watermark(wav, sample_rate=self.sr)).unsqueeze(0) for wav in wavs
        ]
        return result
