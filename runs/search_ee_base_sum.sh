cuda_devices=${cuda_devices:-0}
resume_from=${resume_from:-null}
diff_opt=${diff_opt:-sum}
exp_name=${exp_name:-base_2} # 缩小范围搜索
exp_name=${exp_name}_${diff_opt}

rm -r output/eval/PaSST-EE-${exp_name}
mkdir -p output/eval/PaSST-EE-${exp_name}/res
echo Eval $resume_from >> output/eval/PaSST-EE-${exp_name}/res.txt

# default: 2-6, 0.2-1
# 1: 4, 0.65 0.7 0.75
# 2: 2-6, 1.05 1.1 1.15 1.2 1.5

for patience in {4..6}
# for patience in 4
do 
    # for diff_threshold in 0.2 0.4 0.6 0.8 1.0
    # for diff_threshold in 0.65 0.7 0.75
    for diff_threshold in 1.05 1.1 1.15 1.2 1.5
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