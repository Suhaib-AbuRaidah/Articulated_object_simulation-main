python scripts/generate_data_seg.py --object-set Shape2Motion/faucet/train data/Shape2Motion/faucet_train_1K --num-scenes 1000 --pos-rot 0 --global-scaling 0.6 --num-proc 50 --sample-method mix --dense-photo
python scripts/generate_data_seg.py --object-set Shape2Motion/faucet/val data/Shape2Motion/faucet_val_50 --num-scenes 50 --pos-rot 0 --global-scaling 0.6 --num-proc 25 --sample-method uniform --dense-photo --canonical
python scripts/generate_data_seg_test.py --object-set Shape2Motion/faucet/test data/Shape2Motion/faucet_test_standard --pos-rot 0 --global-scaling 0.6 --dense-photo --mp

python scripts/generate_data_seg.py --object-set Shape2Motion/oven/train data/Shape2Motion/oven_train_1K --num-scenes 1000 --pos-rot 1 --global-scaling 0.5 --num-proc 50 --sample-method mix --dense-photo
python scripts/generate_data_seg.py --object-set Shape2Motion/oven/test data/Shape2Motion/oven_test_50 --num-scenes 50 --pos-rot 1 --global-scaling 0.5 --num-proc 25 --sample-method uniform --dense-photo --canonical
python scripts/scripts/generate_data_seg_test.py --object-set Shape2Motion/oven/test data/Shape2Motion/oven_test_standard --pos-rot 1 --global-scaling 0.5 --dense-photo --mp

python scripts/generate_data_seg.py --object-set Shape2Motion/cabinet/train data/Shape2Motion/cabinet_train_1K --num-scenes 1000 --pos-rot 1 --global-scaling 0.8 --num-proc 50 --sample-method mix --dense-photo
python scripts/generate_data_seg.py --object-set Shape2Motion/cabinet/test data/Shape2Motion/cabinet_test_50 --num-scenes 50 --pos-rot 1 --global-scaling 0.8 --num-proc 25 --sample-method uniform --dense-photo --canonical
python scripts/scripts/generate_data_seg_test.py --object-set Shape2Motion/cabinet/test data/Shape2Motion/cabinet_test_standard --pos-rot 1 --global-scaling 0.8 --dense-photo --mp

python scripts/generate_data_seg.py --object-set Shape2Motion/laptop/train data/Shape2Motion/laptop_train_1K --num-scenes 1000 --pos-rot 0 --global-scaling 0.8 --num-proc 50 --sample-method mix --dense-photo
python scripts/generate_data_seg.py --object-set Shape2Motion/laptop/test data/Shape2Motion/laptop_test_50 --num-scenes 50 --pos-rot 0 --global-scaling 0.8 --num-proc 25 --sample-method uniform --dense-photo --canonical
python scripts/scripts/generate_data_seg_test.py --object-set Shape2Motion/laptop/test data/Shape2Motion/laptop_test_standard --pos-rot 0 --global-scaling 0.8 --dense-photo --mp

python scripts/generate_data_seg.py --object-set syn/cabinet/test data/syn/cabinet_test_syn --num-scenes 1000 --pos-rot 0 --global-scaling 0.6 --num-proc 50 --sample-method mix --dense-photo

python scripts/generate_data_seg.py --object-set syn/example data/syn/example_t1 --num-scenes 10 --pos-rot 0 --global-scaling 0.6 --num-proc 1 --sample-method mix --dense-photo --sim-gui
python scripts/generate_data_seg.py --object-set syn/example1 data/syn/example1_t1 --num-scenes 10 --pos-rot 0 --global-scaling 0.6 --num-proc 1 --sample-method mix --dense-photo --sim-gui
python scripts/generate_data_seg.py --object-set syn/example2 data/syn/example2_t1 --num-scenes 20 --pos-rot 0 --global-scaling 0.6 --num-proc 1 --sample-method mix --dense-photo --sim-gui
python scripts/generate_data_seg2.py --object-set syn/2L1J_Y data/syn/CORR1 --num-scenes 10 --pos-rot 0 --global-scaling 0.6 --num-proc 1 --sample-method mix --dense-photo --sim-gui --rand-state
python scripts/generate_data_seg.py --object-set syn/microwave/train data/syn_local/microwave --num-scenes 10 --pos-rot 0 --global-scaling 0.6 --num-proc 1 --sample-method mix --dense-photo --sim-gui --rand-state
python scripts/generate_data_seg1.py --object-set Shape2Motion/cabinet/eval data/Shape2Motion_local/cabinet_eval --num-scenes 5 --pos-rot 1 --global-scaling 0.6 --num-proc 1 --sample-method mix --dense-photo --sim-gui --rand-state
python scripts/generate_data_seg1.py --object-set Shape2Motion/cabinet/eval data/Shape2Motion_local/cabinet_eval --num-scenes 5 --pos-rot 1 --global-scaling 0.6 --num-proc 1 --sample-method mix --dense-photo --sim-gui --rand-state
python scripts/generate_data_seg5.py --object-set Shape2Motion/robotic_arm/ data/Shape2Motion_gcn/robotic_arm_testing --num-scenes 1 --pos-rot 1 --global-scaling 1 --num-proc 1 --sample-method mix --dense-photo --sim-gui --rand-state
python scripts/generate_data_seg.py --object-set Shape2Motion/robotic_arm/ data/Shape2Motion_coupled/robotic_arm3 --num-scenes 1 --pos-rot 1 --global-scaling 0.8 --num-proc 1 --sample-method mix --dense-photo --rand-state
python scripts/generate_data_seg.py --object-set robotic_arm/ data/Shape2Motion_serial/robotic_arm/val --num-scenes 24 --pos-rot 1 --global-scaling 0.7 --num-proc 1 --sample-method mix --dense-photo --rand-state

python scripts/generate_particulate_pointcloud_data.py data/serial_train \
  --object-set robotic_arm \
  --num-scenes 10 \
  --num-proc 1 \
  --num-points 20000 \
  --rand-state

python scripts/generate_particulate_pointcloud_nocs_data.py data/dataset/robotic_arm \
  --object-set robotic_arm \
  --num-scenes 100 \
  --num-proc 1 \
  --num-points 8000 \
  --rand-state \
  --sample-points
  --max-parts 4 \
  --max-moving-joints 3

python scripts/generate_particulate_pointcloud_nocs_data.py data/dataset/articulated_lamp/train \
 --articraft-root "~/Articulated_object_simulation-main/data/urdfs/Canonical1" \
 --category articulated_task_lamp \
 --num-scenes 20 \
 --num-points 4096

articulated_task_lamp/sub_categories/articulated_task_lamp/fit/train
python scripts/generate_particulate_pointcloud_nocs_world_aligned_data.py data/serial_canonical_aligned_max4 \
  --object-set robotic_arm \
  --num-scenes 20 \
  --num-proc 1 \
  --num-points 8000 \
  --rand-state \
  --max-parts 4 \
  --max-moving-joints 3

python dataset_generation.py data/vlm_articulated \
  --object-set Shape2Motion/robotic_arm \
  --num-scenes 40 \
  --num-views 6 \
  --pos-rot 1 \
  --rand-state \
  --pos-rot 1 \
  --global-scaling 0.9 \
  --num-proc 2

# Create folders and move files for each selected category
for category in \
    "articulated_task_lamp" \
    "xyz_cartesian_stage" \
    "robotic_arms" \
    "serial_elbow_arm" \
    "folding_arm_chain" \
    "extension_ladder" \
    "studio_spotlight_on_yoke" \
    "twostage_telescoping_slide" \
    "cctv_mast_with_pantilt_camera_head" \
    "astronomical_telescope_on_tripod" \
    "orthogonal_xy_stage" \
    "threestage_telescoping_slide" \
    "cantilever_articulated_arm" \
    "desktop_monitor_with_tilt_swivel_stand" \
    "laptop_stand_with_articulated_height_adjustment" \
    "tv_wall_mount" \
    "monitor_mount"; do
    mkdir -p "$category"
    find . -maxdepth 1 -type f -name "rec_${category}_*.tar.gz" -exec mv {} "$category"/ \;
    echo "$(ls $category | wc -l) files moved to $category/"
done

python scripts/generate_particulate_pointcloud_twopose_data.py data/dataset/train \
 --data-type train \
 --category cartesian_stage \
 --category laptop_stand_with_articulated_height_adjustment \
 --category parabolic_dish_on_azimuth_elevation_mount \
 --num-scenes 400 \
 --point-source mesh-sampling \
 --num-points 4096