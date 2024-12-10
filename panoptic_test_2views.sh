#!/bin/bash

dataset_dir=datasets/171026_pose3

# dataset_dir exists, then remove it
if [ -d "$dataset_dir" ]; then
    rm -r $dataset_dir
fi

python src/scripts/convert_to_dtu_rajrup.py /bigdata2/rajrup/datasets/mvsplat_data/171026_pose3_no_ground --output_dir datasets/171026_pose3

sleep 1

python src/scripts/convert_as_dtu_rajrup.py --input_dir datasets/171026_pose3 --output_dir datasets/171026_pose3

sleep 1

python -m src.main +experiment=panoptic checkpointing.load=checkpoints/re10k.ckpt mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_171026_pose3_nctx2_rajrup.json wandb.name=panoptic/views2 test.compute_scores=true