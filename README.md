# code

## preprocessing
download code to Lian_project_code/
 
## train DRIT
cd src

python train_pelvic.py --gpu {GPU_ID} --dataroot {DATA_DIR} --display_dir {LOG_DIR} --checkpoint_dir {CHECKPOINT_DIR}

## test DRIT
python test_pelvic.py --gpu {GPU_ID} --dataroot {DATA_DIR} --checkpoint_dir {CHECKPOINT_DIR} --result_dir {OUTPUT_DIR}


