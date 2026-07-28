# ============================ Dataset Selection ============================
# Generic base config only. Dataset-specific defaults are applied by
# dataset_defaults.apply_dataset_preset().

DATASET_KEY = ""
DATA_DIR = ""
TARGET_COL = None
TARGET_COLS = None
MKT = "dataset"
SEED = 42
DETERMINISTIC = False
# Allow TF32 tensor cores for fp32 matmuls (Ampere+). On an A6000 this roughly doubles the
# matmul peak (38.7 -> 77.4 TFLOP/s) for a 10-bit mantissa on the multiply; the accumulate
# stays fp32. Measured 1.31x on the denoiser forward. It is a numerical change (CRPS moves in
# the 4th-5th decimal), so it defaults to False and DETERMINISTIC=True forces it off.
ALLOW_TF32 = False
VERBOSE = False
DEBUG = False
PIPELINE_PREDS = None
ARTIFACT_ROOT = "./ldt"


# ============================ Data & Preprocessing ============================

PRED = 1
WINDOW = 1
COVERAGE = 0.0
date_batching = True
split_policy = "global_purged_horizon"
split_scope = "global_target_time"
exact_timestamp_batches = True

# Dataset presets set BATCH_SIZE from the paper/table defaults and mirror it
# into DATES_PER_BATCH so the public batch-size knob controls both loader paths.
BATCH_SIZE = 1
DATES_PER_BATCH = 1
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# DataLoader worker settings. The date-batching path uses a fixed _ListBatchSampler with no
# shuffling, and the only per-sample RNG (context missingness) is seeded by sample identity,
# so workers cannot reorder or perturb batches -- which is what DiffusionSplitCache's
# sequential fingerprint alignment requires. 0 (the default) keeps the historical
# single-threaded loader; 8 measured 16.6 -> 5.9 ms/batch on noaa_uk.
DATALOADER_NUM_WORKERS = 0
DATALOADER_PERSISTENT_WORKERS = False
DATALOADER_PREFETCH_FACTOR = None


# ============================ VAE (Set-VAE) ============================

VAE_INPUT_DIM = 2
VAE_OUTPUT_DIM = 1

VAE_LATENT_CHANNELS = 24
VAE_LATENT_DIM = 128
VAE_LAYERS = 3
VAE_HEADS = 4
VAE_FF = 256
VAE_DROPOUT = 0.1
VAE_ENTITY_CONDITION = True
VAE_NUM_ENTITIES = None

VAE_DIR = "./ldt/vae/saved_model/dataset"
_VAE_ENTITY_SUFFIX = "_entity" if VAE_ENTITY_CONDITION else ""
VAE_CKPT = f"{VAE_DIR}/pred-{PRED}_ch-{VAE_LATENT_CHANNELS}{_VAE_ENTITY_SUFFIX}_elbo.pt"

VAE_LEARNING_RATE = 1e-4
VAE_WEIGHT_DECAY = 1e-4
VAE_WARMUP_EPOCHS = 5
VAE_KL_ANNEAL_EPOCHS = 25
VAE_MIN_EPOCHS = 40
VAE_BETA = 1e-3
VAE_MAX_PATIENCE = 20
VAE_INPUT_DROPOUT = 0.20
VAE_NOISE_STD = 0.01
VAE_CONSIST_LAMBDA = 0.0
VAE_RECON_BALANCE = "none"
VAE_PLOT_LATENTS = False
VAE_AMP = False


# ======================= Summarizer (LaplaceAE) =======================

SUM_DIR = "./ldt/summarizer/saved_model/dataset"
SUM_CONTEXT_LEN_FIXED = None
SUM_CONTEXT_LEN = 1
SUM_CONTEXT_DIM = 256
SUM_MIX_DIM = 64
SUM_TV_HIDDEN = 32
SUM_TIME2VEC_DIM = 9
SUM_POS_ENCODING = "learned_abs"
SUM_ROPE_BASE = 10000.0
SUM_DROPOUT = 0.1

SUM_LR = 5e-4
SUM_WEIGHT_DECAY = 1e-4
SUM_EPOCHS = 200
SUM_GRAD_CLIP = 1.0
SUM_AMP = True
SUM_MAX_NONFINITE_GRAD_STEPS = 8
SUM_PATIENCE = 10
SUM_MIN_DELTA = 1e-6
SUM_LOSS_W_X = 1.0
SUM_LOSS_W_V = 0.1
SUM_LOSS_W_T = 0.1
SUM_LOSS_W_DT = 0.0
SUM_LOSS_W_OBS = 0.0
SUM_CHANNEL_BALANCED_X_LOSS = False
SUM_IRREG_POOLING = "none"
SUM_IRREG_HIDDEN = 32
SUM_IRREG_RES_SCALE = 0.1
SUM_T_TOKEN_MODE = "none"
SUM_T_TOKEN_SCALE = 0.1
SUM_FT_MODE = "none"
SUM_FT_LR_MULT = 0.1
SUM_FT_WEIGHT_DECAY = 1e-4
SUM_FT_START_EPOCH = 0

SUM_CKPT = f"{SUM_DIR}/{PRED}-{VAE_LATENT_CHANNELS}-summarizer.pt"


# ============================ Diffusion Model (LLapDiT) ============================

CKPT_DIR = "./ldt/checkpoints/dataset"

TIMESTEPS = 1000
SCHEDULE = "cosine"
PREDICT_TYPE = "v"

LOSS_WEIGHT_SCHEME = "weighted_min_snr"
MINSNR_GAMMA = 5.0
MINSNR_NORMALIZE = "auto"

MODEL_WIDTH = 256
NUM_LAYERS = 5
NUM_HEADS = 4
LAPLACE_K = 256
RHO_CONDITIONING_MODE = "raw"

# Denoiser dynamical core: "lti" (constant poles + residual MLP, original LLapDiff) or
# "chirp" (time-varying poles, stable-by-construction, residual MLP off by default).
DENOISER_MODAL_TYPE = "lti"
# Pole-basis size M. Under CHIRP_BASIS="half_integer" the top basis frequency is M/4
# cycles across the window L (= CHIRP_TIME_SCALE, resolving to PRED), so the Nyquist
# ceiling of a unit-gap grid is M <= 2*L: 24 at h=12, 96 at h=48, 336 at h=168. The old
# default of 256 was ~10x past Nyquist at h=12 and ~2.7x at h=48 -- it only added jitter
# to rho(t)/omega(t) and quadratically many coefficient-head parameters (2*K*M outputs).
# 8 is a safe, horizon-agnostic starting point; sweep it per horizon (the model warns
# when the top frequency exceeds L/2).
CHIRP_NUM_BASIS = 8
CHIRP_RHO_MIN = 1e-4
CHIRP_USE_MLP_RESIDUAL = False
# Window length that normalizes the chirp basis frequencies to the time axis. None
# (default) resolves to the run's horizon (config.PRED) at model-build time — a fixed
# per-run constant, so the function class does not depend on the sample. Set a number to
# pin it explicitly, or the string "adaptive" to opt into the per-sample L = max|t_rel|.
CHIRP_TIME_SCALE = None
# LapFormer output head (skip-scale + LayerNorm/Linear residual): "auto" keeps the
# original head for lti and drops the uncertified residual for chirp (Theorem B);
# "on"/"off" force it either way (2x2 factorial ablation cells).
DENOISER_OUTPUT_HEAD = "auto"
# Theorem-C analytic UQ head (chirp core only, predict_type='x0', certified path):
# per-mode initial variance p0_k and noise intensity q_k -> closed-form latent
# Gaussian law evaluated by a stable 1-D quadrature.
CHIRP_UQ_HEAD = False
# Diffusion training loss: "mse" (default) or "gaussian_nll" (requires CHIRP_UQ_HEAD;
# trains mean and variance jointly for calibrated analytic UQ).
DIFF_LOSS_MODE = "mse"
# Theorem-B' growth budget c_g (chirp core): a learned, capped envelope excursion
# admitting within-window amplitude growth up to a factor e^{c_g}. 0.0 disables the
# head and recovers Theorem B exactly. T2 sweep: {0, log 2, log 5}.
CHIRP_GROWTH_BUDGET = 0.0
# L2 penalty weight on the chirp pole-variation coefficients (Tier-2 ablation):
# shrinks the pole functions toward the constant-pole LTI special case. 0.0 (default)
# disables it; applies to training only (never the val diagnostics), chirp core only.
CHIRP_COEFF_L2 = 0.0
# Pole-function parameterization (chirp core, Phase-4 ablation): "p_exact" (default:
# nonneg Fourier basis with closed-form antiderivative), "p_mono" (monotone integrated
# poles directly; closed-form derivative), or "p_grid" (pointwise positive poles +
# trapezoid integration on the query grid — numerical error grows with gap width,
# the deliberate contrast case).
CHIRP_PARAMETERIZATION = "p_exact"
# Basis for the chirp rho variation. "centered" (default) uses phi-1 = cos, which has
# zero integral over the window, so making rho time-varying adds NO net decay and the
# envelope over the horizon is set purely by the rho floor. "nonneg" is the legacy
# phi = 1+cos expansion, whose mean-1 basis couples rho variation to rho mean and
# extinguishes oscillatory modes when the true decay is small.
CHIRP_RHO_BASIS = "centered"
# Same choice for the omega variation. "centered" (default) uses phi-1 = s*cos, zero mean
# over the window, so a swept frequency does not drag the mean frequency up with it.
# "nonneg" is the legacy 1+s*cos, whose mean-1 basis taxes every unit of variation with an
# equal mean shift and lets a redundant common mode inflate the mean for free (a trained
# half-integer model spent 63% of its omega coefficient mass on that common mode, pushing
# mean omega to 1.23 vs a truth mean of 0.40). Pre-fix checkpoints keep "nonneg".
CHIRP_OMEGA_BASIS = "centered"
# Upper bound on the chirp rho floor as a multiple of 1/horizon (rho_max = scale/PRED),
# the decay analogue of the Nyquist cap on omega: a mode needs rho*H <~ a few to survive
# the forecast window. Sweepable via --chirp-rho-max-scale.
CHIRP_RHO_MAX_SCALE = 4.0
# Harmonic set for the pole basis phi_m(t) = 1 + s_m cos(2 pi f_m t / L). "half_integer"
# (default) uses f = 0.5, 0.5, 1.0, 1.0, ... with alternating signs, so each harmonic is
# available with both signs and the poles can DRIFT across the window. "integer" is the
# legacy f = 1..M with all signs +1: every basis function then completes a whole number of
# cycles, pinning omega(0) = omega(L) = sup_t omega for every mode, so a monotone sweep is
# outside the function class entirely (it fits the H2 chirp truth at only R^2 = 0.35 vs
# 0.99 for half_integer). Pre-fix checkpoints keep "integer".
CHIRP_BASIS = "half_integer"


# ============================ Training Hyperparameters ============================

EPOCHS = 600
BASE_LR = 1.5e-4
MIN_LR = 3e-6
WARMUP_FRAC = 0.095
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 1.0
EARLY_STOP = 20
EARLY_STOP_MIN_EPOCHS = 0

DROPOUT = 0.0
ATTN_DROPOUT = 0.0
DROP_COND_P = 0.18
COND_ADAPTER_MODE = "none"
COND_ADAPTER_HIDDEN = 128
COND_ADAPTER_SCALE = 0.1
COND_ADAPTER_DROPOUT = 0.0
DIFF_INIT_CKPT = None
LR_SCHEDULE = "warmup_constant"
TRAIN_T_SAMPLER = "uniform"
LATENT_NORM_MODE = "global"
COND_TRAIN_MODE = "auto"
IMPUTATION_TRAINING = True
TARGET_MASK_AUX_P = 0.0
TARGET_MASK_AUX_KEEP_MODE = "prefix"
TARGET_MASK_AUX_KEEP_PROB = 0.5
TARGET_MASK_AUX_KEEP_STRIDE = 4
TARGET_MASK_AUX_START_EPOCH = 10

SELF_COND = False
SELF_COND_P = 0.5
SELF_COND_START_EPOCH = 450

# Internal diffusion training acceleration. The public batch-size surface remains
# BATCH_SIZE; these flags only control safe reuse of frozen upstream artifacts.
DIFF_PRECOMPUTE_INPUTS = True
DIFF_PRECOMPUTE_LATENT_DTYPE = "float32"
DIFF_PRECOMPUTE_SUMMARY_DTYPE = "float16"
DIFF_PRECOMPUTE_DIR = None
# Re-derive the cache's per-batch fingerprints by iterating train+val+test at startup, even
# when the cache already exists. That is a full pass over all three splits (~25 s on noaa_uk)
# on EVERY run, to recompute digests the manifest already stores verbatim. Turning it off is
# safe because DiffusionSplitCache._claim asserts the fingerprint of every batch of every
# epoch anyway -- a stale cache then fails loudly at first use instead of being rebuilt.
DIFF_PRECOMPUTE_VERIFY_PLAN = True
# Validate that all entities in a batch share one target query grid. The identical check
# already runs on CPU inside the loader collate, so this is a redundant second pass; keeping
# it costs one device sync per batch.
DIFF_VALIDATE_DT_GRIDS = True


# ============================ Evaluation & Sampling ============================

USE_EMA_EVAL = True
EMA_DECAY = 0.999
PRIMARY_EVAL_METRIC = "crps"
VAL_METRIC_SOURCE = "ema"
TEST_METRIC_SOURCE = "ema"
EMA_COMPARE = True
DIFF_AMP = False
FINAL_TEST_EVAL = "run"
EVAL_EVERY = 1
# Must be > 0 whenever PRIMARY_EVAL_METRIC == "crps": it gates the val CRPS eval that
# feeds checkpoint selection and early stopping. At 0 the trainer computes no CRPS, so
# it saves no best checkpoint and never early-stops (it falls back to the last epoch).
DOWNSTREAM_EVAL_EVERY = 5
# NOTE: EARLY_STOP counts EVALS, not epochs, so it is coupled to DOWNSTREAM_EVAL_EVERY:
# at every=5 with EARLY_STOP=20 the patience is 100 epochs. Raising the interval without
# lowering EARLY_STOP multiplies the patience and makes runs LONGER, not shorter.
# The val diagnostic is a full pass over the val loader; at 1 it ran every single epoch.
VAL_DIAG_EVERY = 5
IRREG_CHECK_EVERY = 0
EMA_COMPARE_EVERY = 0

GEN_STEPS = 64
GEN_ETA = 0.0
IMPUTATION_RANDOM_MASK_RATIO = 0.30

NUM_EVAL_SAMPLES = 25
GUIDANCE_STRENGTH = (1.0, 2.0)
GUIDANCE_POWER = 0.3


# ---------------- In-training validation protocol (selection only) ----------------
# _sampling_kwargs(prefix="EVAL") resolves EVAL_<NAME> before the global fallback, and
# prefix="EVAL" is used at exactly two call sites -- the trainer's val CRPS evals, whose
# ONLY jobs are choosing the best epoch and driving early stopping. Everything that
# produces a reported number (the final test read, llapdiff-checkpoint-eval, the tuning
# harness's eval_sampling.py) goes through prefix="TEST" and is unaffected by these keys.
#
# So this protocol only has to RANK checkpoints the same way the full one does; it does not
# have to reproduce its values. Verify that with a rank-correlation check before relying on
# it (see w_docs/DEVELOPER_GUIDE.md §7.3).
#
# Unset (None / 0) means "use the reported protocol", i.e. the historical behaviour.
EVAL_STEPS = None          # DDIM steps; None -> GEN_STEPS (64)
EVAL_NUM_SAMPLES = None    # ensemble size; None -> NUM_EVAL_SAMPLES (25)
EVAL_MAX_BATCHES = 0       # cap on val batches per eval; 0 -> the whole loader
# How EVAL_MAX_BATCHES picks its subset. "stride" spreads the subset across the whole split
# (the splits are chronological, so a "prefix" subset of a weather series is a season, not a
# sample of one). "prefix" keeps the historical truncation used by --max-eval-batches.
EVAL_SUBSET_MODE = "stride"
# Seed for the val sampler's RNG. None (the historical behaviour) draws from the global torch
# RNG, which makes the val CRPS unpaired across epochs -- fatal once the ensemble is small --
# and lets evaluation perturb the training stream. Setting it makes epochs directly comparable.
EVAL_SEED = None

DYNAMIC_THRESH_P = 0.0
DYNAMIC_THRESH_MAX = 1.0
KARRAS_RHO = 7.5

OUT_DIR = "./ldt/output/dataset"
POLE_PLOT_DIR = f"{OUT_DIR}/pole_plots"


# ============================ Diagnostics ============================

TRAIN_LOSS_T_BINS = 0
LATENT_PROBE_BATCHES = 0
POLE_PROBE = False
POLE_PROBE_EVERY = 5
VAL_DIAG_SNR_BINS = 0
IRREG_CHECK_BATCHES = 4

# Normalization of the history-summary conditioning. "sample" (default) z-scores each
# window over its own sequence axis, so the conditioning is a function of that window
# alone. "batch" is the legacy mode: it z-scores over (batch, sequence), which couples a
# window's conditioning to the OTHER windows in its batch -- and training batches are
# shuffled while eval batches are sequential, so the same window is conditioned
# differently at eval than in training (measured on H2: 41% relative error, cos 0.92).
# That mismatch made conditioning actively harmful (MSE_cond > MSE_uncond).
COND_NORM_MODE = "sample"
