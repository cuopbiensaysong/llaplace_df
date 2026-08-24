#!/usr/bin/env bash
# Latent-width sweep for the two finance cells, then Gate 0 on each result.
#
# WHY. The entity-encode fix (B20) barely moved these cells: crypto 0.617 -> 0.624 (+0.007),
# us_equity 0.643 -> 0.690 (+0.047), both far below the 0.85 gate. B20 predicts a LARGER gain
# on a wider panel -- it measured +0.073 at 4 entities and +0.131 at 40 -- and crypto has 118
# entities per window, us_equity 221. The prediction fails, so permutation invariance is not
# what caps these cells.
#
# The surviving hypothesis is latent capacity, which tracks the measured round-trips closely:
#     noaa_uk    ch 16 / 4 ent   = 4.000 ch per entity -> 0.9952
#     noaa_us    ch 24 / 40 ent  = 0.600              -> 0.800
#     crypto     ch 16 / 118 ent = 0.136              -> 0.624
#     us_equity  ch 12 / 221 ent = 0.054              -> 0.690
# B20 dismissed capacity with a sentence true only of its own cell ("C=16 against N=4 on
# noaa_uk was never the binding constraint"), and --latent-channels exists precisely because
# the preset width was chosen when the encoder could not tell entities apart.
#
# READ. If a wider latent clears 0.85, the cell is usable and we know the width it needs. If
# the curve stays flat near 0.62-0.69, capacity is not the constraint either, and neither cell
# should be taken into diffusion -- a seed costs ~26 h and would be forecasting a latent that
# cannot represent a third of its own target.
#
# The channel count is part of the artifact filename, so nothing here overwrites the widths
# already trained.
set -u
cd "$(dirname "$0")/.." || exit 1

source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
export LLAPDIFF_SRC=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df
export PYTHONPATH=$LLAPDIFF_SRC
RESOLVED=$(python -c 'import llapdiffusion;print(llapdiffusion.__file__)')
case "$RESOLVED" in "$LLAPDIFF_SRC"/*) ;; *) echo "FATAL: $RESOLVED"; exit 2 ;; esac

R=finetuning/results/finance
mkdir -p "$R"
say() { echo "[sweep] $*"; }

one() {   # cell, ch, gpu
  local cell="$1" ch="$2" gpu="$3"
  local tag="${cell}_h100_ch${ch}"
  if [ ! -f "$R/vae_${tag}.done" ]; then
    say "train  $tag on gpu $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -m llapdiffusion.tools.run_stage_pretrain \
      --dataset-key "$cell" --pred 100 --stage vae --entity-encode \
      --latent-channels "$ch" --epochs 80 --seed 0 \
      --json-out "$R/vae_${tag}.json" > "$R/vae_${tag}.log" 2>&1 \
      && touch "$R/vae_${tag}.done"
  else
    say "skip training $tag (already done)"
  fi
  say "gate0  $tag"
  CUDA_VISIBLE_DEVICES="$gpu" python -m llapdiffusion.tools.run_stage1_roundtrip \
    --dataset-key "$cell" --pred 100 --split val --max-batches 200 \
    --vae-entity-encode --latent-channels "$ch" \
    --json-out "$R/gate0_${tag}.json" > "$R/gate0_${tag}.log" 2>&1
  grep -E "^${cell}_h100" "$R/gate0_${tag}.log" | tail -1
}

# Wave 1: three widths in parallel, one per GPU.
one crypto    48 0 &
one crypto    96 1 &
one us_equity 48 2 &
wait
# Wave 2: the last one.
one us_equity 96 0

say "================ summary ================"
python - <<'PY'
import json, pathlib, re
R = pathlib.Path("finetuning/results/finance")
print(f"{'cell':11s} {'ch':>4s} {'ch/ent':>7s} {'round-trip':>11s} {'std_ratio':>10s}  gate@0.85")
ent = {"crypto": 118, "us_equity": 221}
rows = [("crypto", 16), ("crypto", 48), ("crypto", 96),
        ("us_equity", 12), ("us_equity", 48), ("us_equity", 96)]
for cell, ch in rows:
    log = R / (f"gate0_{cell}_h100_ch{ch}.log" if ch not in (16, 12)
               else f"gate0_after_vae.log")
    corr = sd = None
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.startswith(f"{cell}_h100"):
                continue
            # gate0_after_vae.log holds BOTH the tool's own line and a before/after summary
            # row that also starts with the cell name. Taking the first two floats off the
            # summary row reports `shipped` as the correlation and `entenc` as the std ratio.
            # The tool's line is the one with the '-' seed column; require it.
            if " - " not in line and not re.search(r"\s-\s+\d", line):
                continue
            m = re.findall(r"(\d+\.\d+)", line)
            if len(m) >= 2:
                corr, sd = float(m[0]), float(m[1])
    r = ch / ent[cell]
    if corr is None:
        print(f"{cell:11s} {ch:4d} {r:7.3f} {'-':>11s} {'-':>10s}  (missing)")
    else:
        print(f"{cell:11s} {ch:4d} {r:7.3f} {corr:11.3f} {sd:10.2f}  "
              f"{'PASS' if corr >= 0.85 else 'FAIL'}")
PY
say "done"
