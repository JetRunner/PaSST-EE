cuda_devices=${cuda_devices:-1}
resume_from=${resume_from:-null}
diff_opt=${diff_opt:-sigmoid_max}
exp_name=${diff_opt}

rm -r output/eval/PaSST-EE-${exp_name}
mkdir -p output/eval/PaSST-EE-${exp_name}/res
echo Eval $resume_from >> output/eval/PaSST-EE-${exp_name}/res.txt

for patience in {2..6}
do 
    for diff_threshold in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4
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