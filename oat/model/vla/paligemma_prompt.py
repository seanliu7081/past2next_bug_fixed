"""PaliGemma prompt for P2N-VLA: pi05 text format with a discretized state.

The prompt matches openpi ``PaligemmaTokenizer.tokenize`` in its pi05 form
(``src/openpi/models/tokenizer.py``)::

    BOS + encode(f"Task: {clean(instruction)}, State: {' '.join(bins)};\\nAction: ")

with ``clean = strip(), '_' -> ' ', '\\n' -> ' '`` (no lowercasing) and NO EOS. The state is the
prompt-state vector (``state_transforms.PromptStateSpec``) normalized with the train-split q01/q99 to
[-1, 1], clipped, and digitized into 256 bins. Prompts are right-padded with ``<pad>`` (id 0) to
``max_len``; a prompt longer than ``max_len`` raises instead of being truncated.

LIBERO language comes from ``task_uid`` (30..39 for LIBERO-10, ``oat/env/libero/env.py`` order). The zarr
``prompt`` field must never be used: it is stored as ``<U31`` and truncates every instruction (uids 30 and
37 even collide).
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import os
import warnings
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from omegaconf import OmegaConf
from torch import Tensor

from oat.model.vla.state_transforms import (  # noqa: F401  (re-exported prompt-state API)
    LIBERO_PROMPT_STATE, PROMPT_STATE_TRANSFORMS, PromptStateSpec, quat_xyzw_to_axis_angle,
    rot6d_to_axis_angle)


__all__ = [
    "BOS_ID", "DEFAULT_MAX_PROMPT_LEN", "DEFAULT_SPM_PATH", "EOS_ID", "INSTRUCTION_SOURCES", "LIBERO10_INSTRUCTIONS",
    "LIBERO10_INSTRUCTIONS_SOURCE", "LIBERO10_UIDS", "LIBERO_PROMPT_STATE", "N_STATE_BINS", "PAD_ID",
    "PROMPT_STATE_TRANSFORMS", "PaliGemmaTokenizer", "PromptBuilder", "PromptStateSpec", "check_constant_instruction",
    "clean_instruction", "derive_libero10_instructions", "discretize_state", "format_prompt",
    "quat_xyzw_to_axis_angle", "rot6d_to_axis_angle", "uid_from_obs",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPM_PATH = REPO_ROOT / "data/pretrained/p2n_vla/paligemma_tokenizer.model"
PALIGEMMA_VOCAB_SIZE = 257152
PAD_ID, EOS_ID, BOS_ID = 0, 1, 2
N_STATE_BINS = 256
DEFAULT_MAX_PROMPT_LEN = 96
LIBERO10_INSTRUCTION_SOURCE = "libero10"
INSTRUCTION_SOURCES = (LIBERO10_INSTRUCTION_SOURCE,)
LIBERO10_UIDS = tuple(range(30, 40))

# Verified against libero (benchmark 'libero_10', task_order_index 0) in tests/test_p2n_vla_prompt.py.
_LIBERO10_FALLBACK: Dict[int, str] = {
    30: "put both the alphabet soup and the tomato sauce in the basket",
    31: "put both the cream cheese box and the butter in the basket",
    32: "turn on the stove and put the moka pot on it",
    33: "put the black bowl in the bottom drawer of the cabinet and close it",
    34: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    35: "pick up the book and place it in the back compartment of the caddy",
    36: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    37: "put both the alphabet soup and the cream cheese box in the basket",
    38: "put both moka pots on the stove",
    39: "put the yellow and white mug in the microwave and close it",
}


def _libero_config_present() -> bool:
    root = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    return os.path.isfile(os.path.join(root, "config.yaml"))


def derive_libero10_instructions() -> Optional[Dict[int, str]]:
    """LIBERO-10 language keyed by global task uid, exactly as ``LiberoEnv`` assigns them.

    uids enumerate ``libero_task_map`` suites in order (``oat/env/libero/env.py``); the language is
    ``benchmark.get_benchmark_dict()[suite]().get_task(local_id).language``. Returns None when libero is
    not importable, or when its config file is missing (importing ``libero.libero`` would then block on
    ``input()``).
    """
    try:
        if importlib.util.find_spec("libero") is None or not _libero_config_present():
            return None
        with contextlib.redirect_stdout(io.StringIO()):
            from libero.libero import benchmark
            from libero.libero.benchmark.libero_suite_task_map import libero_task_map
            suite = benchmark.get_benchmark_dict()["libero_10"]()
            uid, table = 0, {}
            for suite_name, task_names in libero_task_map.items():
                for local_id, _ in enumerate(task_names):
                    if suite_name == "libero_10":
                        table[uid] = str(suite.get_task(local_id).language)
                    uid += 1
    except Exception as exc:  # an optional dependency must never break importing the prompt module
        warnings.warn(f"Could not derive LIBERO-10 instructions from libero ({exc!r}); using the "
                      "verified fallback table", RuntimeWarning)
        return None
    return table


def _build_libero10_table() -> Tuple[Dict[int, str], str]:
    derived = derive_libero10_instructions()
    if derived is None:
        return dict(_LIBERO10_FALLBACK), "fallback"
    if derived != _LIBERO10_FALLBACK:
        warnings.warn("libero's LIBERO-10 instructions differ from the verified fallback table; using "
                      "libero's (they define the simulator prompts). Prompts on hosts without libero "
                      "will differ.", RuntimeWarning)
    return derived, "libero"


LIBERO10_INSTRUCTIONS, LIBERO10_INSTRUCTIONS_SOURCE = _build_libero10_table()


class PaliGemmaTokenizer:
    """SentencePiece PaliGemma tokenizer: ``<pad>``=0, ``<eos>``=1, ``<bos>``=2, 257152 pieces."""

    pad_id, eos_id, bos_id = PAD_ID, EOS_ID, BOS_ID

    def __init__(self, model_path: str):
        import sentencepiece

        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"PaliGemma SentencePiece model not found: {path}")
        self.model_path = str(path)
        self._processor = sentencepiece.SentencePieceProcessor(model_file=self.model_path)
        found = (self._processor.pad_id(), self._processor.eos_id(), self._processor.bos_id())
        if found != (PAD_ID, EOS_ID, BOS_ID):
            raise ValueError(f"Expected pad/eos/bos ids {(PAD_ID, EOS_ID, BOS_ID)}, got {found}")
        if self._processor.get_piece_size() != PALIGEMMA_VOCAB_SIZE:
            raise ValueError(f"Expected {PALIGEMMA_VOCAB_SIZE} PaliGemma pieces, got "
                             f"{self._processor.get_piece_size()}")

    @property
    def vocab_size(self) -> int:
        return int(self._processor.get_piece_size())

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        return [int(i) for i in self._processor.encode(text, add_bos=bool(add_bos))]

    def decode(self, ids: Sequence[int]) -> str:
        return self._processor.decode([int(i) for i in ids])

    def id_to_piece(self, token_id: int) -> str:
        return self._processor.id_to_piece(int(token_id))


def clean_instruction(text: str) -> str:
    """openpi cleaning: strip, ``'_' -> ' '``, ``'\\n' -> ' '`` (case preserved)."""
    if not isinstance(text, str):
        raise TypeError("instruction must be a str")
    return text.strip().replace("_", " ").replace("\n", " ")


def check_constant_instruction(text: str) -> str:
    """Validate a fixed natural-language instruction (``instruction_source`` that is not a source name).

    Any string other than a source name becomes the prompt of EVERY sample, so a misspelled source
    (``'LIBERO10'``, ``'libero_10'``, ``'libero90'``) would otherwise silently train all tasks on one
    meaningless instruction. Strings that look like LIBERO source names, and single words, are rejected.
    """
    cleaned = clean_instruction(text)
    if not cleaned:
        raise ValueError("A constant instruction must be a nonempty string")
    if "".join(ch for ch in text.lower() if ch.isalnum()).startswith("libero"):
        raise ValueError(f"instruction_source {text!r} looks like a LIBERO source name; the LIBERO language source "
                         f"is spelled exactly {LIBERO10_INSTRUCTION_SOURCE!r} (a constant instruction is natural "
                         "language)")
    if len(cleaned.split()) < 2:
        raise ValueError(f"instruction_source {text!r} is neither a known source {list(INSTRUCTION_SOURCES)} nor a "
                         "natural-language instruction (a constant instruction needs at least two words)")
    return text


def format_prompt(instruction: str, bins: Sequence[int]) -> str:
    """The pi05 prompt text for one sample (state bins are integers in 0..255)."""
    values = []
    for value in bins:
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= int(value) < N_STATE_BINS:
            raise ValueError(f"State bins must be integers in [0, {N_STATE_BINS}), got {value!r}")
        values.append(str(int(value)))
    return f"Task: {clean_instruction(instruction)}, State: {' '.join(values)};\nAction: "


def discretize_state(x_norm: Tensor) -> Tensor:
    """256-bin pi05 state discretization: ``np.digitize(clip(x, -1, 1), linspace(-1, 1, 257)[:-1]) - 1``.

    ``x_norm`` is the q01/q99-normalized state ``[B, D]`` (any float dtype/device). Returns int64 in 0..255.
    """
    if not isinstance(x_norm, torch.Tensor) or not x_norm.is_floating_point():
        raise TypeError("x_norm must be a floating torch.Tensor")
    if x_norm.ndim < 1:
        raise ValueError("x_norm must have at least one dimension")
    if not torch.isfinite(x_norm).all():
        raise ValueError("Normalized prompt state must be finite")
    with torch.autocast(device_type=x_norm.device.type, enabled=False):
        boundaries = torch.linspace(-1.0, 1.0, N_STATE_BINS + 1, dtype=torch.float64,
                                    device=x_norm.device)[:-1]
        clipped = x_norm.double().clamp(-1.0, 1.0)
        # np.digitize (right=False) puts x in bin i with b[i-1] <= x < b[i]: torch.bucketize(right=True).
        return torch.bucketize(clipped, boundaries, right=True) - 1


def uid_from_obs(obs: Mapping[str, Tensor]) -> Tensor:
    """Per-sample task uid ``[B]`` int64 from ``obs['task_uid']`` of shape ``[B,To,1]``, ``[B,To]``/``[B,1]`` or ``[B]``.

    Training batches carry int64 and rollouts float (the runner casts to the policy dtype); floats are rounded
    and must be integral to within 1e-3. The last frame is used.
    """
    if not isinstance(obs, Mapping) or "task_uid" not in obs:
        raise KeyError("Observations must contain 'task_uid'")
    value = obs["task_uid"]
    if not isinstance(value, torch.Tensor):
        raise TypeError("task_uid must be a torch.Tensor")
    if value.dtype == torch.bool or value.is_complex():
        raise TypeError(f"task_uid must be integer or float, got {value.dtype}")
    if value.ndim == 3:
        if value.shape[-1] != 1:
            raise ValueError(f"task_uid [B,To,K] must have K == 1, got {tuple(value.shape)}")
        value = value[:, -1, 0]
    elif value.ndim == 2:
        value = value[:, -1]
    elif value.ndim != 1:
        raise ValueError(f"task_uid must have shape [B,To,1], [B,To] or [B], got {tuple(value.shape)}")
    if value.is_floating_point():
        value64 = value.double()
        if not torch.isfinite(value64).all():
            raise ValueError("task_uid must be finite")
        rounded = value64.round()
        if ((value64 - rounded).abs() > 1e-3).any():
            raise ValueError("task_uid must contain integer task ids (got non-integral values; normalized?)")
        return rounded.long()
    return value.long()


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


def _as_uid(key) -> int:
    if isinstance(key, bool):
        raise ValueError(f"Instruction uids must be integers, got {key!r}")
    if isinstance(key, Integral):
        return int(key)
    if isinstance(key, str) and key.strip().lstrip("-").isdigit():
        return int(key.strip())
    raise ValueError(f"Instruction uids must be integers, got {key!r}")


class PromptBuilder:
    """Token ids for a batch of prompts.

    ``instruction_source``:
    - ``'libero10'`` (exact spelling): ``LIBERO10_INSTRUCTIONS`` by task uid (uids required; unknown uids raise);
    - any other string: one constant natural-language instruction (uids ignored, ``build`` accepts
      ``uids=None``); single words and LIBERO-like source names are rejected (``check_constant_instruction``);
    - a mapping ``{uid: instruction}``: per-uid instructions (uids required; unknown uids raise).
    """

    def __init__(self, tokenizer, instruction_source: Union[str, Mapping[int, str]],
                 max_len: int = DEFAULT_MAX_PROMPT_LEN):
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("tokenizer must provide encode(text, add_bos=True)")
        if isinstance(max_len, bool) or not isinstance(max_len, Integral) or max_len < 2:
            raise ValueError("max_len must be an integer >= 2")
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.pad_id = int(getattr(tokenizer, "pad_id", PAD_ID))
        source = _plain(instruction_source)
        self.instruction_source = source
        self.constant_instruction: Optional[str] = None
        self.instructions: Dict[int, str] = {}
        if isinstance(source, str):
            if source == LIBERO10_INSTRUCTION_SOURCE:
                self.instructions = dict(LIBERO10_INSTRUCTIONS)
            else:
                self.constant_instruction = check_constant_instruction(source)
        elif isinstance(source, Mapping):
            for key, text in source.items():
                uid = _as_uid(key)
                if not isinstance(text, str) or not clean_instruction(text):
                    raise ValueError(f"Instruction for uid {uid} must be a nonempty string")
                if uid in self.instructions:
                    raise ValueError(f"Duplicate instruction uid {uid}")
                self.instructions[uid] = text
            if not self.instructions:
                raise ValueError("An instruction mapping must not be empty")
        else:
            raise TypeError("instruction_source must be 'libero10', an instruction string, or {uid: instruction}")

    @property
    def requires_uids(self) -> bool:
        return self.constant_instruction is None

    def instruction(self, uid: Optional[int] = None) -> str:
        if self.constant_instruction is not None:
            return self.constant_instruction
        if uid is None:
            raise ValueError("This instruction source maps task uids to instructions; uids are required")
        uid = int(uid)
        if uid not in self.instructions:
            raise ValueError(f"Unknown task uid {uid}; known uids: {sorted(self.instructions)}")
        return self.instructions[uid]

    def prompt_text(self, uid: Optional[int], bins: Sequence[int]) -> str:
        return format_prompt(self.instruction(uid), bins)

    def encode(self, uid: Optional[int], bins: Sequence[int]) -> List[int]:
        text = self.prompt_text(uid, bins)
        ids = self.tokenizer.encode(text, add_bos=True)
        if len(ids) > self.max_len:
            raise ValueError(f"Prompt has {len(ids)} tokens > max_len {self.max_len}: {text!r}")
        return ids

    def worst_case_lengths(self, n_state_dims: int) -> Dict[Optional[int], int]:
        """Token count per known instruction with every bin at 3 digits (each digit is one piece)."""
        bins = [N_STATE_BINS - 1] * int(n_state_dims)
        if self.constant_instruction is not None:
            return {None: len(self.tokenizer.encode(format_prompt(self.constant_instruction, bins), add_bos=True))}
        return {uid: len(self.tokenizer.encode(format_prompt(text, bins), add_bos=True))
                for uid, text in sorted(self.instructions.items())}

    def build(self, uids: Optional[Tensor], state_bins: Tensor) -> Tuple[Tensor, Tensor]:
        """``(ids [B, max_len] int64, valid [B, max_len] bool)`` on CPU, right-padded with ``<pad>``."""
        if not isinstance(state_bins, torch.Tensor) or state_bins.ndim != 2:
            raise ValueError("state_bins must be a [B, D] tensor")
        if state_bins.dtype == torch.bool or state_bins.is_floating_point() or state_bins.is_complex():
            raise TypeError(f"state_bins must be an integer tensor, got {state_bins.dtype}")
        bins = state_bins.detach().cpu().long()
        if bins.numel() and (bins.min() < 0 or bins.max() >= N_STATE_BINS):
            raise ValueError(f"state_bins must lie in [0, {N_STATE_BINS})")
        batch = bins.shape[0]
        uid_list: List[Optional[int]] = [None] * batch
        if self.requires_uids:
            if uids is None:
                raise ValueError("This instruction source needs per-sample task uids")
            if not isinstance(uids, torch.Tensor):
                uids = torch.as_tensor(uids)
            if uids.dtype == torch.bool or uids.is_floating_point() or uids.is_complex():
                raise TypeError(f"uids must be an integer tensor, got {uids.dtype}")
            if tuple(uids.shape) != (batch,):
                raise ValueError(f"uids must have shape [{batch}], got {tuple(uids.shape)}")
            uid_list = [int(u) for u in uids.detach().cpu().tolist()]
        rows = [self.encode(uid, row) for uid, row in zip(uid_list, bins.tolist())]
        ids = torch.full((batch, self.max_len), self.pad_id, dtype=torch.long)
        valid = torch.zeros((batch, self.max_len), dtype=torch.bool)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
            valid[index, :len(row)] = True
        return ids, valid
