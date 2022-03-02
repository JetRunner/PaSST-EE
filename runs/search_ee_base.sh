cuda_devices=${cuda_devices:-1}
resume_from=${resume_from:-"/nas/Algorithm_Engineering/zc/PaSST-EE/output/PaSST-EE-base-2gpu/audioset/_2/checkpoints/epoch=113-step=472616.ckpt"}
diff_opt=${diff_opt:-max}
exp_name=${exp_name:-base}
exp_name=${exp_name}_${diff_opt}


for patience in {2..6}
do 
    for diff_threshold in 0.05 0.1 0.2 0.3 0.4 0.5
    do
        CUDA_VISIBLE_DEVICES=${cuda_devices} python ex_audioset.py evaluate_only \
            with trainer.precision=16 \
                trainer.resume_from_checkpoint=${resume_from} \
                datasets.test.batch_size=1 \
                patience=${patience} \
                diff_threshold=${diff_threshold} \
                diff_opt=${diff_opt} \
                models.net.arch=passt_deit_bd_p16_384 \
                -p -F "output/eval/PaSST-EE-${exp_name}/logs"
    done
done