#!/usr/bin/env bash

set -x

EXP_DIR=exps/two_stage/deformable-detr-hybrid-branch/36eps/vit/vit_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --epochs 75 \
    --lr_backbone 1e-4 \
    --lr_backbone_decay_rate 0.8 \
    --wd_backbone 0.0001 \
    --lr_drop 60 \
    --num_queries_one2one 300 \
    --num_queries_one2many 1500 \
    --k_one2many 6 \
    --lambda_one2many 1.0 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    --backbone vit_large \
    --pretrained_backbone_path pretrained_backbone/mae_pretrain_vit_large.pth \
    --use_checkpoint \
    --use_fp16 \
    ${PY_ARGS}
