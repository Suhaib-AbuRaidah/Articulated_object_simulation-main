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