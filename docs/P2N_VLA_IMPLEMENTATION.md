# P2N-VLA implementation contract

This is the interface contract for the P2N-VLA implementation. The approved design is in
`/root/.claude/plans/give-me-a-plan-concurrent-knuth.md`, which is the source of truth for intent.
This file is the source of truth for **names, signatures, tensor shapes and dtypes**. Every new
module must follow it exactly, so that modules written in parallel fit together.

Ground rules:
- **New files only.** Never modify an existing repository file.
- Python is `/venv/oat/bin/python` (torch 2.10, transformers 5.2.0, sentencepiece 0.2.2). Do not
  upgrade packages in `/venv/oat`.
- Match the house style of `oat/policy/p2n_new_common.py`: validate inputs eagerly and raise on bad input.
- Never call `torch.inference_mode()` in training-reachable code. Use `torch.no_grad()`.
  Never lazily memoize tensors on modules: build them per call or register them as buffers in `__init__`.
- Tests live in `tests/test_p2n_vla_*.py`. They must run on CPU with tiny configs.
  - Mark tests that need real assets with `@pytest.mark.requires_pi05` and make them `pytest.skip` loudly when the assets are missing.
  - Mark CUDA-only tests with `@pytest.mark.gpu`.
  - Register both markers in `tests/conftest_p2n_vla.py`, which `tests/test_p2n_vla_*.py` imports, or via `pytest_configure` in a new `tests/conftest.py`. Do not edit any existing conftest.

## Assets (M0, done)

| Asset | Location |
|---|---|
| pi05 weights | `/workspace/.hf_home/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/model.safetensors` (F32, 812 tensors, sha256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`) |
| PaliGemma SentencePiece | `data/pretrained/p2n_vla/paligemma_tokenizer.model` (sha256 `8986bb4f…fc6`) |
| Manifest | `data/pretrained/p2n_vla/assets.json` |
| OAT (LIBERO, SO(3) left_noise, EMA) | `/workspace/hf_upload/tokenizer_oattok_so3aug_ep4960_mse0.001.ckpt` |
| LIBERO data | `/workspace/past_action/data/libero/libero10_N500.zarr` (use this absolute path) |

pi05 key prefix: `paligemma_with_expert.`

| Checkpoint keys | Tensors |
|---|---|
| `paligemma.model.vision_tower.vision_model.*` | SigLIP |
| `paligemma.model.multi_modal_projector.linear.{weight,bias}` | projector |
| `paligemma.model.language_model.layers.N.{input_layernorm,post_attention_layernorm}.weight` | VLM layer norms |
| `paligemma.model.language_model.layers.N.self_attn.{q,k,v,o}_proj.weight` | VLM attention |
| `paligemma.model.language_model.layers.N.mlp.{gate,up,down}_proj.weight` | VLM MLP |
| `paligemma.model.language_model.norm.weight` | VLM final norm |
| `paligemma.lm_head.weight` | tied embedding [257152,2048] |
| `gemma_expert.model.layers.N.{input_layernorm,post_attention_layernorm}.dense.{weight,bias}` | expert adaRMS [3072,1024]+[3072] |
| `gemma_expert.model.layers.N.self_attn.*_proj.weight`, `gemma_expert.model.layers.N.mlp.*_proj.weight` | expert attention and MLP |
| `gemma_expert.model.norm.dense.{weight,bias}` | expert final adaRMS |
| `gemma_expert.lm_head.weight` | **always dropped** |

Top-level (no prefix): `action_in_proj.{weight,bias}` [1024,32], `action_out_proj.{weight,bias}` [32,1024], `time_mlp_in.{weight,bias}`, `time_mlp_out.{weight,bias}` [1024,1024]. Verify every name against the header with `safetensors.safe_open`; do not trust this table blindly.

## Shared numerical conventions

- `NEG = -2.3819763e38`. This is the additive mask value. Never use `-inf` in masks. Every query row keeps its own diagonal key, so no row is ever fully masked.
- **RMSNorm (Gemma):** `y = x̂·(1+w)`, with `x̂ = x·rsqrt(mean(x²)+eps)` computed in fp32, eps 1e-6. Cast the output back to the input dtype.
- **adaRMS:** `[scale|shift|gate] = chunk(Dense(c), 3)`, `y = x̂·(1+scale)+shift`. The residual is `x + gate·sublayer(y)`, using the raw gate.
  - This applies to both the attention and MLP sublayers.
  - The final expert norm returns only `y` (its gate rows are zero in the checkpoint and are discarded).
- **RoPE:** θ = 10000, rotate-half over the full head_dim.
  - `inv_freq = θ^(-arange(0,hd,2)/hd)`, `freqs = pos·inv_freq`, `emb = cat(freqs, freqs)`.
  - `x' = x·cos + rotate_half(x)·sin`, where `rotate_half(x) = cat(-x[..., hd/2:], x[..., :hd/2])`.
  - Compute in fp32 per call and cast back.
- **Attention scale:** q·256^-0.5 (head_dim^-0.5).
- **Embeddings:** text embeddings ×√width_vlm. SigLIP+projector image tokens are unscaled. KI rows ×√width_vlm. AR `tok_emb` ×32, i.e. √1024 for the full expert and √width_expert in general.
- **GeGLU:** `down(gelu_tanh(gate(x)) · up(x))`.
- **Time embedding (pi05):** `sincos(t, dim=1024, min_period=4e-3, max_period=4.0)`.
  - `fraction = linspace(0,1,dim/2)`, `period = min_period·(max_period/min_period)^fraction`.
  - `arg = 2π·t/period`, output `cat(sin(arg), cos(arg))` (sin first).
  - `cond = silu(time_mlp_out(silu(time_mlp_in(sincos))))`.
- **Dtypes:**
  - Frozen pi05 backbone weights are bf16 `nn.Parameter(requires_grad=False)`: SigLIP, projector, Gemma-2B linear and embedding weights, and LoRA base layers.
  - All norm weights and everything else are fp32.
  - The policy owns autocast: `torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type=='cuda', cache_enabled=False)`.
  - Tiny/CPU configs run entirely in fp32. Each builder takes `frozen_dtype`, defaulting to bf16 for the full size and fp32 for tiny.

## `oat/model/vla/specs.py`

```python
@dataclass(frozen=True)
class GemmaSpec:  width:int; depth:int; mlp_dim:int; num_heads:int; num_kv_heads:int; head_dim:int
                  vocab_size:Optional[int]=None; eps:float=1e-6; rope_theta:float=10000.0
GEMMA_2B   = GemmaSpec(2048, 18, 16384, 8, 1, 256, vocab_size=257152)
GEMMA_300M = GemmaSpec(1024, 18, 4096, 8, 1, 256)
TINY_VLM    = GemmaSpec(64, 2, 128, 8, 1, 16, vocab_size=257152)   # full vocab so real token ids work
TINY_EXPERT = GemmaSpec(32, 2, 64, 8, 1, 16)                        # same depth as TINY_VLM (required)
@dataclass(frozen=True)
class SiglipSpec: hidden:int; intermediate:int; layers:int; heads:int; patch:int; image_size:int; eps:float=1e-6
                  @property num_tokens -> (image_size//patch)**2
SIGLIP_SO400M = SiglipSpec(1152, 4304, 27, 16, 14, 224)
TINY_SIGLIP   = SiglipSpec(32, 64, 2, 4, 14, 28)                    # 4 tokens/image
MODEL_SIZES = {'full': (SIGLIP_SO400M, GEMMA_2B, GEMMA_300M), 'tiny': (TINY_SIGLIP, TINY_VLM, TINY_EXPERT)}
```

The VLM and expert must share depth, num_heads, num_kv_heads and head_dim. Assert this.

## `oat/model/vla/siglip.py`

```python
class SiglipEncoder(nn.Module):
    def __init__(self, spec: SiglipSpec, out_dim: int, frozen_dtype=torch.bfloat16)
        self.vision = transformers.SiglipVisionModel(SiglipVisionConfig(hidden_size=..., intermediate_size=...,
                 num_hidden_layers=..., num_attention_heads=..., patch_size=..., image_size=..., layer_norm_eps=eps,
                 hidden_act='gelu_pytorch_tanh', vision_use_head=False))   # keys: vision.vision_model.*
        self.projector = nn.Linear(spec.hidden, out_dim)                     # keys: projector.{weight,bias}
        # all params requires_grad_(False); module always eval (override train())
    def forward(self, pixels: Tensor[N,3,S,S] in [-1,1]) -> Tensor[N, num_tokens, out_dim]   # runs under torch.no_grad()
```

Assert that the instantiated model has no `head` submodule.

## `oat/model/vla/lora.py`

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float)
    # state keys: base.weight (frozen), lora_A [rank,in] fp32 (kaiming_uniform a=√5), lora_B [out,rank] fp32 zeros
    def forward(self, x): return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * (alpha/rank)
def inject_lora(module: nn.Module, rank: int, alpha: float,
                targets=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')) -> list[str]
# Replaces matching nn.Linear children anywhere under `module`. It is called on joint.vlm.layers ONLY,
# after base weights are loaded. Returns the replaced module names.
```

## `oat/model/vla/layout.py` (masks and positions; pure functions, no parameters)

```python
SEG_RAWDIFF, SEG_HIST, SEG_AR = 0, 1, 2
N_RAWDIFF_SLOTS = 9          # 7 raw + acc + jerk
HIST_POSITION_OFFSET = 9      # all HIST tokens share pos0+9
AR_POSITION_OFFSET = 10       # AR token k at pos0+10+k

@dataclass(frozen=True)
class PrefixLayout:
    positions: LongTensor[B,P]      # P = nonki_len + ki_len
    bias: Tensor[B,1,P,P] fp32      # 0 / NEG
    valid: BoolTensor[B,P]          # image/prompt validity (KI rows always valid)
    pos0: LongTensor[B]             # number of valid NON-KI tokens
    nonki_len: int; ki_len: int

def build_prefix_layout(img_valid: BoolTensor[B,Ni], prompt_valid: BoolTensor[B,Lp], ki_len: int = 0) -> PrefixLayout
```

`build_prefix_layout` rules:
- Physical order is `[images | prompt (right-padded) | KI]`.
- **Non-KI rows** see every valid non-KI key, plus their own diagonal. Pad rows therefore still see the valid keys. Non-KI rows never see KI keys.
- **KI rows** see every valid non-KI key, plus KI keys causally (`j ≤ i`).
- **Non-KI positions:** `cumsum(valid_nonki) - 1`, clamped at ≥ 0.
- **KI positions:** `pos0 + arange(ki_len)`.

```python
@dataclass(frozen=True)
class SuffixLayout:
    positions: LongTensor[B,S]
    bias: Tensor[B,1,S, nonki_len + S] fp32   # columns: [prefix nonki keys | suffix keys]
    seg_ids: LongTensor[S]
    n_rawdiff: int; n_hist: int; n_ar: int; nonki_len: int

def build_suffix_layout(pos0: LongTensor[B], prefix_valid: BoolTensor[B,nonki_len],
                        rawdiff_valid: Optional[BoolTensor[B,9]], n_hist: int, n_ar: int,
                        log_gate: Optional[Tensor[B,1]] = None, hist_closed: bool = False) -> SuffixLayout

def decode_view(layout: SuffixLayout, ar_index: int) -> tuple[Tensor[B,1,1,K_k], LongTensor[B,1]]
```

`build_suffix_layout` rules:
- Order is `[RAWDIFF (0 or 9) | HIST (0 or 4) | AR (n_ar ≤ 8)]`. RAWDIFF is absent when `rawdiff_valid is None` (the no-past ablation).
- **Positions:** RAWDIFF j → `pos0 + j`; HIST → `pos0 + 9`; AR k → `pos0 + 10 + k`. These never depend on validity or on the variant.
- **RAWDIFF and HIST rows:** diagonal only (0); everything else NEG.
- **AR row i:**
  - valid prefix keys → 0, prefix pads → NEG;
  - RAWDIFF keys → 0 if valid, else NEG;
  - HIST keys → `log_gate` (fp32, per sample), or 0 if `log_gate is None`, or NEG if `hist_closed`;
  - AR keys → 0 for `j ≤ i`, else NEG.

`decode_view` returns the bias row of AR token `ar_index` restricted to the keys that exist at that decode step (`nonki_len + n_rawdiff + n_hist + ar_index + 1` columns), plus its position.

**Invariant (test it):** slicing teacher-forcing rows equals the decode views.

```python
def build_flow_suffix_layout(pos0, prefix_valid, n_act: int) -> SuffixLayout
```

For the flow baseline: positions are `pos0 + arange(n_act)`. Action rows see valid prefix keys and all action keys (bidirectional). `seg_ids` are all `SEG_AR`.

## `oat/model/vla/gemma_joint.py`

### Module tree (state_dict names; the loader maps onto these exactly)

```
GemmaJoint
  vlm: GemmaStack
    embed_tokens: nn.Embedding(vocab, Wv)         # tied output head = embed_tokens.weight
    layers[i]: GemmaLayer
      input_layernorm: GemmaRMSNorm   (.weight)    # fp32
      self_attn: {q_proj,k_proj,v_proj,o_proj}: nn.Linear(bias=False)   # LoRA later wraps → .base.weight, .lora_A, .lora_B
      post_attention_layernorm: GemmaRMSNorm
      mlp: {gate_proj, up_proj, down_proj}
    norm: GemmaRMSNorm
  expert: ExpertStack
    layers[i]: ExpertLayer
      input_layernorm, post_attention_layernorm: AdaRMSNorm(.dense: Linear(We,3We))      [norm_mode='adarms']
                                               | SegmentRMSNorm(.modulation: Param[n_seg,3We]) [norm_mode='segment']
      self_attn: {q_proj,k_proj,v_proj,o_proj}; mlp: {gate_proj, up_proj, down_proj}
    norm: AdaRMSNorm | SegmentRMSNorm   (final; gate ignored)
```

### Signatures

```python
class GemmaJoint(nn.Module):
    def __init__(self, vlm: GemmaSpec, expert: GemmaSpec, norm_mode: Literal['adarms','segment'],
                 n_segments: int = 3, frozen_dtype=torch.bfloat16, activation_checkpointing: bool = True)
    def embed_text(self, ids: LongTensor[B,L]) -> Tensor[B,L,Wv]          # embed_tokens(ids) * sqrt(Wv)
    def prefix_forward(self, embeds: Tensor[B,P,Wv], layout: PrefixLayout) -> PrefixOutput
        # PrefixOutput(hidden: Tensor[B,P,Wv] (after final norm),
        #              kv: tuple[tuple[Tensor[B,1,nonki_len,hd], Tensor[B,1,nonki_len,hd]], ...] per layer (post-RoPE),
        #              pos0: LongTensor[B])
        # VLM attention: SDPA with k/v .expand(B,H,L,hd) (no copy), enable_gqa=False, attn_mask=bias.to(q.dtype).
        # Per-layer torch.utils.checkpoint(use_reentrant=False) when activation_checkpointing and training and grad enabled.
    def expert_forward(self, x: Tensor[B,S,We], prefix_kv, layout_bias: Tensor[B,1,S,K], positions: LongTensor[B,S], *,
                       seg_ids: Optional[LongTensor[S]] = None,   # norm_mode='segment'
                       cond: Optional[Tensor[B,We]] = None,       # norm_mode='adarms'
                       cache: Optional[ExpertCache] = None, use_cache: bool = False,
                       probe_cols: Optional[tuple[int,int]] = None) -> ExpertOutput
        # ExpertOutput(hidden: Tensor[B,S,We] after final norm, cache: Optional[ExpertCache],
        #              probe_mass: Optional[Tensor[n_layers]] = mean softmax mass of the LAST query row on key cols [a,b))
        # Keys/values per layer = cat(prefix_kv[l] (detached by caller), cache.kv[l] if cache, new k/v).
        # K = nonki_len + cache_len + S must equal layout_bias.shape[-1].
        # Expert attention is explicit fp32: logits=(q.float()*scale)@k.float().T + bias.float(); softmax fp32; @v.float(); cast back.

@dataclass(frozen=True)
class ExpertCache: kv: tuple[tuple[Tensor[B,1,L,hd], Tensor[B,1,L,hd]], ...]; length: int   # never mutated; extend returns new
```

## `oat/model/vla/pi05_checkpoint.py`

```python
PI05_REVISION = 'b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba'
def read_header(path) -> dict[str, tuple[str, list[int]]]
def posemb_sincos(t: Tensor[B], dim: int, min_period=4e-3, max_period=4.0) -> Tensor[B,dim]   # fp32, sin first
def compute_c0(state: Mapping[str,Tensor], t0: float, width: int) -> Tensor[width] fp32
def load_pi05(path, *, siglip: SiglipEncoder, joint: GemmaJoint, mode: Literal['flow','p2n'], t0: float = 0.6,
              flow_heads: Optional[nn.ModuleDict] = None,   # keys action_in_proj, action_out_proj, time_mlp_in, time_mlp_out
              strict: bool = True) -> LoadReport
# LoadReport(consumed: list[str], dropped: list[str], c0: Optional[Tensor], sha256: Optional[str])
```

`load_pi05` rules:
- Stream tensors with `safe_open`, casting each to the destination parameter's dtype.
- `paligemma.lm_head.weight` → `joint.vlm.embed_tokens.weight`.
- The loader must be called **before** `inject_lora`.
- **mode='flow':** the joint must have `norm_mode='adarms'`. Copy `dense` weights. `flow_heads` must be provided and are loaded. Drop only `gemma_expert.lm_head.weight`.
- **mode='p2n':** the joint must have `norm_mode='segment'`.
  - `c0 = compute_c0(..., t0)`.
  - For every expert norm, `m = dense.weight @ c0 + dense.bias`, broadcast to every segment row of `modulation`.
  - Drop `gemma_expert.lm_head.weight`, `action_in_proj.*`, `action_out_proj.*`, `time_mlp_in.*` and `time_mlp_out.*`, after `c0` has been computed.
- Assert that the final-norm gate rows (`norm.dense` rows 2W:3W) are all zero.
- With `strict=True`, any unconsumed key that is not allow-listed raises, and any destination parameter left unfilled raises.

## `oat/model/vla/paligemma_prompt.py`, `state_transforms.py`, `oat_ki.py`, `image_preprocess.py`

```python
class PaliGemmaTokenizer:            # sentencepiece; pad_id=0, eos_id=1, bos_id=2
    def __init__(self, model_path: str); def encode(self, text: str, add_bos: bool = True) -> list[int]
LIBERO10_INSTRUCTIONS: dict[int,str]  # uid 30..39 → full LIBERO-10 language; built at import from libero_task_map when available,
                                      # with a hard-coded fallback table verified against libero in tests
def quat_xyzw_to_axis_angle(q: Tensor[...,4]) -> Tensor[...,3]   # exactly robosuite quat2axisangle (clip w, den≈0 → 0)
def rot6d_to_axis_angle(r: Tensor[...,6], layout: Literal['rows','columns']) -> Tensor[...,3]
class PromptStateSpec: keys: list[str]; transforms: dict[str, str]   # e.g. {'robot0_eef_quat': 'quat_to_axis_angle'}
    def extract(self, obs: Mapping[str,Tensor]) -> Tensor[B,D] fp32   # uses obs[k][:, -1] (last frame)
def discretize_state(x_norm: Tensor[B,D]) -> LongTensor[B,D]   # clip to [-1,1]; bucketize(boundaries=linspace(-1,1,257)[:-1], right=True)-1 → 0..255
class PromptBuilder:
    def __init__(self, tokenizer, instruction_source: Union[str, Mapping[int,str]], max_len: int = 96)
    def build(self, uids: Optional[LongTensor[B]], state_bins: LongTensor[B,D]) -> tuple[LongTensor[B,max_len], BoolTensor[B,max_len]]
    # text = f"Task: {clean(instr)}, State: {' '.join(bins)};\nAction: "; clean = strip, '_'→' ', '\n'→' '; add_bos; right-pad 0; raise if too long
def uid_from_obs(obs) -> LongTensor[B]   # obs['task_uid'][:, -1].reshape(B,-1)[:, 0].round().long()

class OATKITable(nn.Module):        # rows: Parameter[5001, Wv] fp32 (row 5000 = KI_BOS)
    def __init__(self, width: int, n_codes: int = 5000, skip: int = 1152)
    def init_from_embedding(self, embed_weight: Tensor[V,Wv]) -> None   # rows[t] = E[V-1-skip-t]; rows[5000] = E[2] (raw rows, no scaling)
    def embed_block(self, targets: LongTensor[B,8]) -> Tensor[B,8,Wv]   # ids=[5000, t0..t6]; rows[ids]*sqrt(Wv)
    def logits(self, hidden: Tensor[B,8,Wv]) -> Tensor[B,8,5000]        # hidden @ rows[:5000].T  (fp32)

class ImagePreprocessor:
    def __init__(self, rgb_ports: list[str], wrist_ports: list[str], image_size: int = 224)
    def __call__(self, obs, *, train_aug: bool, generator: Optional[torch.Generator] = None) -> Tensor[B, n_cams, 3, S, S] in [-1,1]
    # input obs[port] [B,To,H,W,3] uint8 or float 0..255 → last frame; train_aug: non-wrist 95% random crop (side sqrt(0.95)) + resize back + rotate U(-5,5)°;
    # all cams brightness ±0.3 contrast ±0.4 saturation ±0.5 (openpi semantics, in [0,1] space, clamp); then resize to S (bilinear, antialias) → x*2-1
```

## `oat/dataset/vla_dataset.py`

```python
class PromptStateStatsMixin:   # prepended to RealRobotZarrDatasetWithPrevWindow / RealRobotZarrDatasetWithStateHistory
    def __init__(self, *args, prompt_state: Mapping, **kwargs)   # prompt_state = {'keys': [...], 'transforms': {...}}
    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer
        # super().get_normalizer() + field 'prompt_state' = SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats)
        # with q01/q99 computed over training frames (normalization_train_mask) of the TRANSFORMED prompt-state vector:
        # scale = 2/(q99-q01+1e-6), offset = -1 - q01*scale. All its params requires_grad_(False).
class VLAZarrDatasetWithPrevWindow(PromptStateStatsMixin, RealRobotZarrDatasetWithPrevWindow)
class VLAZarrDatasetWithStateHistory(PromptStateStatsMixin, RealRobotZarrDatasetWithStateHistory)
```

## Policy (M3): `oat/policy/p2n_vla_common.py` and its siblings

`P2NVLACommonPolicy(P2NNewCommonPolicy)` calls `BasePolicy.__init__(self)` directly, as `P2NLatentFlowCommonPolicy` does.

**Submodule names, which also serve as artifact keys:**
- `siglip`, `joint`, `ki_table`;
- `raw_proj`, `acc_proj`, `jerk_proj`, `type_embedding`, `action_time_embedding`, `tok_emb`;
- `latent_adapter` (which owns the frozen OAT);
- `action_normalizer`;
- buffers `_self_past_optimizer_step`, `_context_schema`, `_variant_code`, `_normalizer_fitted`.

**Subclasses:**
- `P2NVLAPolicy`
- `P2NVLAStateGatePolicy`: adds `history_encoder`, `summary_type_embedding`, `observation_pool`, `history_gate`.
- `PI05KIFlowPolicy`: adds `flow_heads`, and uses `norm_mode='adarms'`.

**Frozen base keys** (excluded from artifacts):
- `siglip.*`;
- `joint.vlm.embed_tokens.weight`;
- `joint.vlm.layers.*.{self_attn,mlp}.*.base.weight`;
- `joint.vlm.layers.*.*layernorm.weight`;
- `joint.vlm.norm.weight`.

The full method contract is specified in the M3 workflow prompt.

---

## M1 core: as implemented (authoritative public API)

The core has been implemented and tested:
- `tests/test_p2n_vla_core.py` (16 tests);
- `tests/test_p2n_vla_loader.py` (7 tests, including real-weight CPU and GPU loads).

Do **not** change these public signatures. Fixes must keep them.

**`layout.py`**
- `NEG`, `SEG_RAWDIFF=0`, `SEG_HIST=1`, `SEG_AR=2`, `N_SEGMENTS=3`, `N_RAWDIFF_SLOTS=9`, `HIST_POSITION_OFFSET=9`, `AR_POSITION_OFFSET=10`.
- `build_prefix_layout(img_valid[B,Ni], prompt_valid[B,Lp], ki_len=0) -> PrefixLayout(positions, bias[B,1,P,P], valid, pos0[B], nonki_len, ki_len)`.
  - Property `.nonki_valid` → `[B, nonki_len]`.
- `build_suffix_layout(pos0, prefix_valid[B,nonki_len], rawdiff_valid[B,9] | None, n_hist, n_ar, log_gate[B,1] | None = None, hist_closed=False) -> SuffixLayout(positions, bias[B,1,S,nonki_len+S], seg_ids[S], n_rawdiff, n_hist, n_ar, nonki_len)`.
  - Properties `.n_cond` and `.hist_columns`, the latter giving the `(start, stop)` key columns of HIST.
- `prefill_view(layout, n_rows) -> (bias, positions, seg_ids)`.
- `decode_view(layout, ar_index) -> (bias_row[B,1,1,K], position[B,1])`.
- `build_flow_suffix_layout(pos0, prefix_valid, n_act) -> SuffixLayout`.

**`gemma_joint.py`**
- `GemmaJoint(vlm_spec, expert_spec, norm_mode, n_segments=3, frozen_dtype=bf16, activation_checkpointing=True)`.
  - `.embed_text(ids[B,L]) -> [B,L,Wv]`, scaled by `sqrt(Wv)`.
  - `.vlm_logits(hidden) -> fp32 [B,L,V]`.
  - `.prefix_forward(embeds, PrefixLayout) -> PrefixOutput(hidden[B,P,Wv], kv: per layer (k,v)[B,1,nonki_len,hd], pos0[B])`.
  - `.expert_forward(x[B,S,We], prefix_kv, layout_bias[B,1,S,K], positions[B,S], *, seg_ids=None, cond=None, cache=None, use_cache=False, probe_cols=None) -> ExpertOutput(hidden[B,S,We], cache: ExpertCache|None, probe_mass[n_layers]|None)`.
    - `prefix_kv` must be detached by the caller.
    - In segment mode, `seg_ids` is required.
    - In adarms mode, `cond[B,We]` is required.
- Attributes: `.vlm_spec`, `.expert_spec`, `.norm_mode`.
- `.vlm.embed_tokens.weight` is the tied embedding (bf16, frozen).
- Expert norm modules are `SegmentRMSNorm` (`.modulation[n_seg, 3W]`) or `AdaRMSNorm` (`.dense`).

**`lora.py`**
- `inject_lora(module, rank, alpha, targets=...) -> list[str]`.
- `lora_parameters(module) -> list[Parameter]`.
- `LoRALinear` (`.base`, `.lora_A`, `.lora_B`, `.merged_weight()`).

**`siglip.py`**
- `SiglipEncoder(spec, out_dim, frozen_dtype)`: `forward(pixels[N,3,S,S] in [-1,1]) -> [N,T,out_dim]`.
- Runs under `no_grad`, always in eval, with all parameters frozen.

**`pi05_checkpoint.py`**
- `default_pi05_path()`, `PI05_REVISION`, `PI05_SHA256`, `read_header(path)`.
- `compute_c0(state, t0, width)`.
- `load_pi05(path, *, siglip, joint, mode='flow'|'p2n', t0=0.6, flow_heads=None, strict=True, expected_sha256=None, verify_sha256=False) -> LoadReport(mode, consumed, dropped, folded, c0, t0, sha256)`.
  - Call it **before** `inject_lora`.

**`flow_head.py`**
- `FlowHeads(width, action_dim=32)` (`.action_in_proj`, `.action_out_proj`, `.time_mlp_in`, `.time_mlp_out`).
- `posemb_sincos(t, dim)`.
- `flow_condition(heads, t[B]) -> [B,We]`.
- `flow_velocity(joint, heads, prefix_kv, pos0, prefix_valid, x_t[B,H,32], t[B]) -> [B,H,32]`.
- `sample_actions(joint, heads, prefix_kv, pos0, prefix_valid, noise, num_steps=10)`.

## M3: policy API (implemented by the lead engineer; the M4 workspace builds against it)

### Classes

| Class | Module | `VARIANT` | `VARIANT_CODE` |
|---|---|---|---|
| `P2NVLAPolicy` | `oat.policy.p2n_vla` | `p2n_vla` | 10 |
| `P2NVLAStateGatePolicy` | `oat.policy.p2n_vla_state_gate` | `p2n_vla_state_gate` | 11 |
| `PI05KIFlowPolicy` | `oat.policy.pi05_ki_flow` | `pi05_ki_flow` | 12 |

All three derive from `oat.policy.p2n_vla_common.P2NVLACommonPolicy(P2NNewCommonPolicy)`.

### Capability flags (class attributes)

- `requires_execution_acknowledgement=True`
- `supports_explicit_past_actions=True`
- `supports_explicit_past_action_valid=True`
- `supports_generated_history_validation=True`; for `pi05_ki_flow` this is `False`
- `requires_state_history`: `True` only for the gate variant
- `supports_history_summary_gate`: `True` only for the gate variant
- `policy_family='p2n_vla'`

### Constructor

The constructor uses Hydra keyword arguments. The common ones:

```
shape_meta, n_action_steps=8, n_obs_steps=1, past_n=7, horizon=16, variant=None, task='libero',
construction_mode='fresh'|'restore', model_size='full'|'tiny', pi05_weights=None, pi05_sha256=None,
spm_path=None, spm_sha256=None, tokenizer_checkpoint=None, tokenizer_config=None, tokenizer_metadata=None,
rgb_ports=('agentview_rgb','robot0_eye_in_hand_rgb'), wrist_ports=('robot0_eye_in_hand_rgb',), train_image_aug=True,
prompt={'instruction_source': 'libero10'|str|{uid: str}, 'state_keys': [...], 'state_transforms': {...},
        'max_len': 96, 'dummy_uid': 30},
lora_rank=16, lora_alpha=16.0, lambda_ki=1.0, ki_skip=1152, use_past=True, adarms_t0=0.6,
temperature=0.0, topk=10, activation_checkpointing=True, frozen_dtype=None,
self_past_p=0.5, self_past_warmup_steps=1000, self_past_ramp_steps=4000, self_past_chunk_size=4,
self_past_temperature=1.0, self_past_topk=10, self_past_schedule='optimizer_step'
```

Extra arguments for the gate variant:

```
state_history_steps=8, state_history_keys=None, history_embed_dim=128, history_n_heads=4, history_n_layers=2,
history_summary_tokens=4, history_dropout=0.1, history_gate_mode='learned'|'open'|'closed',
history_gate_hidden_dim=128, history_gate_init=0.9, rotation_6d_layout='rows', gate_state_dim=256
```

### Methods the workspace may call

| Call | Behavior |
|---|---|
| `policy.set_normalizer(dataset.get_normalizer())` | Required before `forward` or `predict_action`. Needs the fields `action`, the prompt-state keys, any history keys, and `prompt_state`. |
| `loss = policy(batch)` | `forward(batch, history_mode=None)`. In training mode, `None` means `'configured'` (self-past by schedule). In eval mode it means `'expert'`. `history_mode='generated'` forces p=1. The `batch` is the dataset sample collated: `obs`, `action`, `past_action`, `past_action_valid`, `prev_obs`, `prev_past_action`, `prev_past_action_valid`, `prev_window_valid`, `episode_step`. Returns a scalar fp32 loss. The policy opens its own autocast, so the caller must NOT wrap it in `accelerator.autocast()`; wrapping is harmless but redundant. |
| `policy.last_loss_components` | A dict of Python floats or `None`. Keys: `loss`, `loss_ar` (or `loss_flow` for the flow baseline), `loss_ki`, `ar_token_acc`, `ki_token_acc`, `self_past_p`, `self_past_rows`, `gate_mean`, `gate_min`, `gate_max`, `hist_attention_mass`. |
| `policy.get_optimizer(policy_lr=5e-5, new_module_lr=1e-4, weight_decay=1e-10, betas=(0.9,0.95), eps=1e-8, fused=None) -> torch.optim.AdamW` | Each param group has `name` ∈ {`pretrained`, `new`}. `fused=None` means fused iff every parameter is on CUDA. Build the optimizer **after** moving the policy to its device. |
| `policy.clip_groups() -> {'ar': [Parameter], 'ki': [Parameter]}` | Disjoint, and together they cover every trainable parameter. `'ki'` may be empty when `lambda_ki == 0`. |
| `policy.on_optimizer_step()` | Advances the self-past counter in training mode. |
| `policy.self_past_step` (property), `policy.set_self_past_step(n)`, `policy.self_past_probability()` | Self-past schedule accessors. |
| `policy.predict_action(obs, past_actions=..., past_action_valid=...)` | Stateless. Returns `{'action':[B,8,7], 'action_pred':[B,16,7]}` and is used for reconstruction MSE. The stateful runner path uses `reset`, `predict_action` and `record_executed_actions`. |
| `policy.get_history_gate_metrics() -> dict` | Gate variant only; any other variant returns `{}`. |
| `policy.trainable_named_parameters() -> list[(name, Parameter)]` | De-duplicated. Used by the EMA. |
| `policy.artifact_state_dict(trainable_override: dict[name, Tensor] \| None = None) -> dict[str, Tensor]` | The state dict minus the frozen base keys. With an override (EMA shadows by name), substitutes those values. |
| `policy.frozen_base_keys() -> list[str]` | The frozen pi05 backbone keys excluded from artifacts. |
| `policy.load_artifact_state(state: dict)` | Checks key accounting and loads. |
| `policy.export_config() -> dict` | Constructor kwargs, including `_target_` and `construction_mode='restore'`. |
| `policy.artifact_metadata() -> dict` | Provenance and schema metadata. |
| `Class.from_checkpoint(path, *, base_weights=None, weights='ema'\|'model', device='cpu', spm_path=None) -> policy` | `base_weights=None` means `default_pi05_path()`. |

### Checkpoint payload

Owned by the workspace. Written atomically with `torch.save`:

```
{'format': 'p2n_vla_checkpoint_v1', 'cfg': OmegaConf container, 'policy_config': policy.export_config(),
 'metadata': policy.artifact_metadata(), 'state_dicts': {'model': artifact (live), 'ema_model': artifact (EMA) or absent},
 'training': {...optimizer/scheduler/ema/rng/counters...} (resume checkpoints only)}
```

A snapshot has `state_dicts = {'ema_model': EMA artifact, with trainable keys in bf16}` and no `training` block.

`from_checkpoint` reads `policy_config` and `state_dicts[weights]`.

## M4: workspace, EMA, launcher, eval and configs

**`oat/model/common/trainable_ema.py`**

```python
class TrainableEMA:
    def __init__(self, named_parameters, decay=0.999, warmup_power=None)   # fp32 shadow per param name
    @torch.no_grad() def step(self, named_parameters)   # shadow = d*shadow + (1-d)*param ; increments self.updates
    def state_dict(self) -> {'decay','updates','shadow': {name: Tensor}} ; def load_state_dict(self, sd)
    @contextlib.contextmanager def swap_in(self, policy)  # copy shadow -> params (stash), yield, restore params AND
                                                        # every module's .training flag; never inside inference_mode
    def trainable_override(self) -> dict[str, Tensor]
```

**`oat/workspace/train_p2n_vla.py`: `TrainP2NVLAWorkspace`**

Derive it from `oat/workspace/train_p2n_new.py`. Reuse its structure and helpers: atomic save, per-rank RNG capture/restore, `_cap_dataloader`, logging. The differences are listed below.

- **Accelerate and DDP:**
  - `Accelerator(mixed_precision='no')` (the policy owns autocast), gradient accumulation, and `DistributedDataParallelKwargs(find_unused_parameters=False, gradient_as_bucket_view=True)`.
  - Call `optimizer.zero_grad(set_to_none=False)`.
- **Setup order:** instantiate the policy → `.to(device)` → `set_normalizer` → `get_optimizer` → `accelerator.prepare(policy, optimizer, train_loader, val_loader)`.
- **LR scheduler:** a custom `LambdaLR` on optimizer steps: linear warmup over `lr_warmup_steps`, then cosine to `min_lr_ratio` (0.1) at `max_optimizer_steps`. It is used for all param groups.
- **Clipping:** separate clipping per `policy.clip_groups()` with `max_grad_norm` each. Log `grad_norm_ar` and `grad_norm_ki`.
- **After each successful optimizer step:** `scheduler.step()`, `policy.on_optimizer_step()`, `ema.step()`, and step logging.
- **Validation** (every `val_every` epochs, on all ranks):
  - Run inside `ema.swap_in(policy)` with `policy.eval()`.
  - Run under `torch.no_grad()`, **not** `inference_mode`.
  - Compute expert-history and generated-history losses, capped by `max_val_steps`.
  - Compute reconstruction MSE through the stateless `predict_action`.
  - Reduce across ranks.
- **Checkpoints:**
  - Every `checkpoint_every` epochs, write `checkpoints/latest.ckpt` (resume payload).
  - Every `snapshot_every` optimizer steps, write `snapshots/upd-{step:06d}_ema.ckpt`.
  - At the end, write a final snapshot and delete `latest.ckpt` when `training.keep_resume_checkpoint` is false.
- **Training length:** `training.max_optimizer_steps` hard-stops training. `max_train_steps` caps micro-batches per epoch.
- **Resume:** requires the same world size, variant and data config.
- **Logging:** W&B (mode from config) plus `logs.jsonl` with every scalar from `last_loss_components`, LR per group, grad norms, `samples_per_sec` and `max_memory_reserved`.

**Configs**
- `oat/config/train_p2n_vla.yaml`: the full recipe; it holds every hyperparameter in the plan.
- `oat/config/train_p2n_vla_state_gate.yaml`
- `oat/config/train_pi05_ki_flow.yaml`

These configs use the task configs `oat/config/task/policy/libero/libero10_vla{,_state_history}.yaml`.

**Launcher: `scripts/train_p2n_vla.py` + `train_p2n_vla.sh`**

`--variant {p2n_vla,p2n_vla_state_gate,pi05_ki_flow} --task libero --tokenizer PATH --pi05 PATH --spm PATH --output DIR --devices 0,1 --num-processes 2 --resume CKPT [--dry-run|--preflight|--probe] -- hydra overrides`

| Mode | What it does |
|---|---|
| `--dry-run` | Composes and validates the config only. |
| `--preflight` | Checks the dataset schema and split, OAT provenance (`scripts/train_p2n_latent_flow.py` `validate_tokenizer_source`), asset sha256s and prompt lengths, builds the model on CPU, and reports parameter counts. |
| `--probe` | `torchrun` with N optimizer steps, self-past forced to p=0.5, reporting memory and samples/s. |
| (no flag) | Launches `torchrun`. |

**`scripts/evaluate_p2n_vla.py`**
- Loads a snapshot or checkpoint with `from_checkpoint`.
- Builds the runner from the embedded task cfg: `P2NNewLiberoRunner` or `P2NStateGateNewLiberoRunner`.
- Options: `--protocol corrected|official`, `--n-test`, `--n-parallel-envs`, `--seed`, `--episode-start-seed`, `--force-gate open|closed`, `--use-k-tokens`, `--temperature`, `--device`.
- Reuses helpers from `scripts/evaluate_candidate.py`: the schedule, Wilson summaries and source hashing.
- Writes `summary.json`, `episodes.jsonl`, `metadata.json` and `source_hashes.json`.

**`docs/P2N_VLA.md`:** usage documentation with literal commands.
