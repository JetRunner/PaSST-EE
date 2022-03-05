cuda_devices=${cuda_devices:-1}
resume_from=${resume_from:-"/nas/Algorithm_Engineering/zc/PaSST-EE/output/PaSST-EE-base-2gpu/audioset/_2/checkpoints/epoch=113-step=472616.ckpt"}
diff_opt=${diff_opt:-sigmoid_max}
exp_name=${exp_name:-base}
exp_name=${exp_name}_${diff_opt}

CUDA_VISIBLE_DEVICES=${cuda_devices} python ex_audioset.py evaluate_only \
    with trainer.precision=16 \
        trainer.resume_from_checkpoint=${resume_from} \
        diff_opt=${diff_opt} \
        datasets.test.batch_size=1 \
        models.net.arch=passt_deit_bd_p16_384 \
        -p