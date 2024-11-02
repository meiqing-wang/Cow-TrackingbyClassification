#!/bin/bash

input_dir=$1
output_dir=$2
txt_inference=$3
vid_name=$4

echo $vid_name

cd "$(dirname "$0")"
script_dir="$(pwd)"
output_dir="$(echo "$script_dir$output_dir")"

echo ""
echo "START OF MASTER SHELL SCRIPT"
echo "input_dir:      $input_dir"
echo "output_dir:     $output_dir"
echo "inference_file: $(echo "$script_dir$output_dir$txt_inference")"

python /mnt/wks2/boris/dev/compute_metrics/cvat_gt_to_eval_input.py\
    --label_dir "$input_dir"\
    --output "$(echo "$output_dir""/0_gt_$vid_name.txt")"

python /mnt/wks2/boris/dev/compute_metrics/remove_id2.py\
    --inp "$(echo "$output_dir""/0_gt_$vid_name.txt")"\
    --out "$(echo "$output_dir""/1_gt_$vid_name.txt")"

python /mnt/wks2/boris/dev/compute_metrics/adapt_objid_gt_tr.py\
    --gt "$(echo "$output_dir""/1_gt_$vid_name.txt")"\
    --tr "$(echo $output_dir$txt_inference)"\
    --output_gt "$(echo "$output_dir""/2_gt_$vid_name.txt")"\
    --output_tr "$(echo "$output_dir""/2_tr_$vid_name.txt")"

python /mnt/wks2/boris/dev/compute_metrics/run_custom_clearmot.py\
    --gt "$(echo "$output_dir""/2_gt_$vid_name.txt")"\
    --tr "$(echo "$output_dir""/2_tr_$vid_name.txt")"





# HOW TO CALL THIS MASTER SHELL SCRIPT
# sh master_shell_eval.sh
#     path_to_labels
#     eval_folder # where to store results
#     inference_file
#     video_name

# EXAMPLE
# sh master_shell_eval.sh \
#     /home/dalco/boris/dev_test/custom_model/data/dataset1/labels/vid95_1 \
#     /eval2 \
#     /tr_eval_inference.txt \
#     vid95_1
