ngpu=${ngpu:-2}
cuda_devices=${cuda_devices:-0,1}
exp_name=${exp_name:-distil}
distillation_alpha=${distillation_alpha:-0.9}

if [[ $ngpu -eq 1 ]]; then
    python ex_audioset_oracle_distillation.py with trainer.precision=16 distillation_alpha=${distillation_alpha} trainer.default_root_dir="output/PaSST-EE-${exp_name}-1gpu" models.net.arch=passt_deit_bd_p16_384 -p -F "output/PaSST-EE-${exp_name}-1gpu/audioset/logs" -c "PaSST EE ${exp_name}"
else
    CUDA_VISIBLE_DEVICES=$cuda_devices DDP=${ngpu} python ex_audioset_oracle_distillation.py with trainer.precision=16 distillation_alpha=${distillation_alpha} trainer.default_root_dir="output/PaSST-EE-${exp_name}-${ngpu}gpu" models.net.arch=passt_deit_bd_p16_384 -p -F "output/PaSST-EE-${exp_name}-${ngpu}gpu/audioset/logs" -c "PaSST ${exp_name} ${ngpu} GPU"
fi