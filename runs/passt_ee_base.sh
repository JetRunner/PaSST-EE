ngpu=${ngpu:-2}
cuda_devices=${cuda_devices:-0,1}

if [[ $ngpu -eq 1 ]]; then
    python ex_audioset.py with trainer.precision=16  models.net.arch=passt_deit_bd_p16_384 -p -F "output/audioset/PaSST-EE-base-1gpu" -c "PaSST EE base"
else
    CUDA_VISIBLE_DEVICES=$cuda_devices DDP=${ngpu} python ex_audioset.py with trainer.precision=16  models.net.arch=passt_deit_bd_p16_384 -p -F "output/audioset/PaSST-EE-base-${ngpu}gpu" -c "PaSST base ${ngpu} GPU"
fi