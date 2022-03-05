cuda_devices=${cuda_devices:-0}
resume_from=${resume_from:-null}
diff_opt=${diff_opt:-softmax_max}
exp_name=${diff_opt}

rm -r output/eval/PaSST-EE-${exp_name}
mkdir -p output/eval/PaSST-EE-${exp_name}/res
echo Eval $resume_from >> output/eval/PaSST-EE-${exp_name}/res.txt

for patience in {2..6}
do 
    for diff_threshold in 0.005 0.01 0.02 0.03 0.04 0.05
    do
        echo [patience $patience, diff_threshold, $diff_threshold] >> output/eval/PaSST-EE-${exp_name}/res.txt
        CUDA_VISIBLE_DEVICES=${cuda_devices} python ex_audioset.py evaluate_only \
            with trainer.precision=16 \
                trainer.resume_from_checkpoint=${resume_from} \
                datasets.test.batch_size=1 \
                patience=${patience} \
                diff_threshold=${diff_threshold} \
                diff_opt=${diff_opt} \
                models.net.arch=passt_deit_bd_p16_384 \
                -p -F "output/eval/PaSST-EE-${exp_name}/logs" > output/eval/PaSST-EE-${exp_name}/res/p${patience}_t${diff_threshold}
        tail -n 3 output/eval/PaSST-EE-${exp_name}/res/p${patience}_t${diff_threshold} >> output/eval/PaSST-EE-${exp_name}/res.txt
        echo -------------------------------------------------------------------------------------- >> output/eval/PaSST-EE-${exp_name}/res.txt
    done
done