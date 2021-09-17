python test.py ../data/work_dirs/cascade_rcnn_r50_40e_DetectoRS/cascade_rcnn_r50_40e_DetectoRS.py \
    ../data/work_dirs/cascade_rcnn_r50_40e_DetectoRS/epoch_25.pth \
    --eval bbox

# python test.py ../data/work_dirs/cascade_rcnn_r50_1x_DetectoRS/cascade_rcnn_r50_1x_DetectoRS.py \
#     ../data/work_dirs/cascade_rcnn_r50_1x_DetectoRS/latest.pth \
#     --format-only \
#     --eval-options "jsonfile_prefix=../data/results/DtetctoRS_b_results"

# python json2submit.py \
#     --test_json ../data/results/DtetctoRS_b_results.bbox.json