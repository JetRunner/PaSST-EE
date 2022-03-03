cuda_devices=${cuda_devices:-0}
exp_name=${exp_name:-base_fix_layer}
resume_from=${resume_from:-/nas/Algorithm_Engineering/zc/PaSST-EE/output/PaSST-EE-base-2gpu/audioset/_2/checkpoints/epoch=113-step=472616.ckpt}

# rm -r output/eval/PaSST-EE-${exp_name}
# mkdir -p output/eval/PaSST-EE-${exp_name}/res
# echo Eval $resume_from >> output/eval/PaSST-EE-${exp_name}/res.txt

for fix_layer in {3..6}
do 
    
    echo [fix_layer as $fix_layer] >> output/eval/PaSST-EE-${exp_name}/res.txt
    CUDA_VISIBLE_DEVICES=${cuda_devices} python ex_audioset.py evaluate_fix_layer \
        with trainer.precision=16 \
            trainer.resume_from_checkpoint=${resume_from} \
            datasets.test.batch_size=1 \
            fix_ic_output_layer_num=${fix_layer} \
            models.net.arch=passt_deit_bd_p16_384 \
            -p -F "output/eval/PaSST-EE-${exp_name}/logs" > output/eval/PaSST-EE-${exp_name}/res/fix_layer_${fix_layer}
    tail -n 3 output/eval/PaSST-EE-${exp_name}/res/fix_layer_${fix_layer} >> output/eval/PaSST-EE-${exp_name}/res.txt
    echo -------------------------------------------------------------------------------------- >> output/eval/PaSST-EE-${exp_name}/res.txt
done