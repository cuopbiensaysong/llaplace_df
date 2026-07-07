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


# ============================ Evaluation & Sampling ============================

USE_EMA_EVAL = True
EMA_DECAY = 0.999
PRIMARY_EVAL_METRIC = "val_diag_mse_raw"
VAL_METRIC_SOURCE = "raw"
TEST_METRIC_SOURCE = "raw"
EMA_COMPARE = True
DIFF_AMP = False
FINAL_TEST_EVAL = "run"
EVAL_EVERY = 1
DOWNSTREAM_EVAL_EVERY = 0
VAL_DIAG_EVERY = 1
IRREG_CHECK_EVERY = 0
EMA_COMPARE_EVERY = 0

GEN_STEPS = 64
GEN_ETA = 0.0
IMPUTATION_RANDOM_MASK_RATIO = 0.30

NUM_EVAL_SAMPLES = 25
GUIDANCE_STRENGTH = (1.0, 2.0)
GUIDANCE_POWER = 1.0

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
