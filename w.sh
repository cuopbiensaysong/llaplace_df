SRC=ldt/checkpoints/llapdiffusion_longest_horizon_artifacts/checkpoints
# dataset:horizon:channels
for spec in bms_air:168:24 uci_air:168:16 physionet:12:16 \
            noaa_us:168:24 noaa_uk:168:16 us_equity:100:12 crypto:100:16; do
  ds=${spec%%:*}; rest=${spec#*:}; H=${rest%%:*}; C=${rest##*:}
  mkdir -p "ldt/vae/saved_model/$ds" "ldt/summarizer/saved_model/$ds"
  mv "$SRC/${ds}_h${H}_vae_best_elbo.pt"   "ldt/vae/saved_model/$ds/pred-${H}_ch-${C}_entity_elbo.pt"
  mv "$SRC/${ds}_h${H}_vae_best_recon.pt"  "ldt/vae/saved_model/$ds/pred-${H}_ch-${C}_entity_recon.pt"
  mv "$SRC/${ds}_h${H}_summarizer_best.pt" "ldt/summarizer/saved_model/$ds/${H}-${C}-summarizer.pt"
done