cd src/

python -u inference.py \
    --test_annotation /opt/ml/input/data/annotations/test.json \
    --test_zip_frames /opt/ml/input/data/zip_frames/test/ \
    --ckpt_file save/model.bin
