#!/usr/bin/env bash
# Wait for the entity-encoded stage-1 VAE trainings to finish, then re-run Gate 0 with
# --vae-entity-encode so the before/after pair is measured on the SAME split with the SAME
# batch cap. The "before" numbers (shipped VAE, val, --max-batches 200) are:
#     crypto    h100  corr 0.617  std_ratio 0.80  n=10112 (6 constant windows skipped)
#     us_equity h100  corr 0.643  std_ratio 0.68  n=15912
# Gate threshold 0.85. For calibration, noaa_uk went 0.9222 -> 0.9952 and noaa_us 0.669 -> 0.800
# under the same fix, and B20 predicts a LARGER gain on a wider panel -- crypto has 118
# entities/window and us_equity 221, against noaa_us's 40. That prediction is what this tests.
set -u
cd "$(dirname "$0")/.." || exit 1

source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
export LLAPDIFF_SRC=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df
export PYTHONPATH=$LLAPDIFF_SRC

RESOLVED=$(python -c 'import llapdiffusion;print(llapdiffusion.__file__)')
case "$RESOLVED" in
  "$LLAPDIFF_SRC"/*) ;;
  *) echo "FATAL: llapdiffusion resolved to $RESOLVED"; exit 2 ;;
esac

R=finetuning/results/finance
GPU="${GPU:-2}"

say() { echo "[gate0-after] $*"; }

say "waiting for run_stage_pretrain to finish..."
while pgrep -f "run_stage_pretrain .*--stage vae" >/dev/null 2>&1; do sleep 120; done
say "stage-1 trainings finished at $(date -Iseconds)"

run_one() {   # cell, pred, expected artifact
  local cell="$1" pred="$2" art="$3"
  if [ ! -f "$art" ]; then
    say "MISSING artifact for ${cell}: $art -- did the training fail? skipping"
    tail -20 "$R/vae_${cell}_h${pred}_entenc.log" 2>/dev/null
    return 1
  fi
  say "=== Gate 0, entity-encoded: ${cell} h${pred} ==="
  CUDA_VISIBLE_DEVICES="$GPU" python -m llapdiffusion.tools.run_stage1_roundtrip \
    --dataset-key "$cell" --pred "$pred" --split val --max-batches 200 \
    --vae-entity-encode \
    --json-out "$R/gate0_${cell}_h${pred}_entenc.json" 2>&1 | tail -8
}

run_one crypto    100 ldt/vae/saved_model/crypto/pred-100_ch-16_entity_entenc_elbo.pt
run_one us_equity 100 ldt/vae/saved_model/us_equity/pred-100_ch-12_entity_entenc_elbo.pt

say "=== before / after ==="
python - <<'PY'
import json, pathlib
R = pathlib.Path("finetuning/results/finance")
rows = []
for cell, pred in (("crypto", 100), ("us_equity", 100)):
    got = {}
    for tag in ("shipped", "entenc"):
        p = R / f"gate0_{cell}_h{pred}_{tag}.json"
        if not p.exists():
            got[tag] = None
            continue
        d = json.loads(p.read_text())
        # the tool nests its per-cell record; find the first dict carrying a correlation
        def find(o):
            if isinstance(o, dict):
                for k in ("corr", "correlation", "roundtrip_corr"):
                    if k in o and isinstance(o[k], (int, float)):
                        return o
                for v in o.values():
                    r = find(v)
                    if r: return r
            elif isinstance(o, list):
                for v in o:
                    r = find(v)
                    if r: return r
            return None
        got[tag] = find(d)
    rows.append((cell, pred, got))

print(f"{'cell':16s} {'shipped':>9s} {'entenc':>9s} {'delta':>9s}  gate@0.85")
for cell, pred, got in rows:
    def val(t):
        r = got.get(t)
        if not r: return None
        for k in ("corr", "correlation", "roundtrip_corr"):
            if k in r: return float(r[k])
        return None
    a, b = val("shipped"), val("entenc")
    if a is None or b is None:
        print(f"{cell+'_h'+str(pred):16s} {'?' if a is None else f'{a:.3f}':>9s} "
              f"{'?' if b is None else f'{b:.3f}':>9s} {'—':>9s}  (incomplete)")
    else:
        print(f"{cell+'_h'+str(pred):16s} {a:9.3f} {b:9.3f} {b-a:+9.3f}  "
              f"{'PASS' if b >= 0.85 else 'FAIL'}")
PY
say "done"
