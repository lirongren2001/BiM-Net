
INPUT_DIM=1190 
NUM_NODES_WM=48
NUM_NODES_GM=82 
DATAPATH='data/HCP'
DEVICE='cuda:0'
EPOCHS=100
BATCH_SIZE=128
LR=0.001
REPETITIONS=10
FOLDS=10
RESULTPATH='results/HCP.txt'
SAVE_PATH='ckpt/HCP/'
SEED=123
WEIGHT_DECAY=0.001
LAYER=1
DROPOUT=0.4
HIDDEN_DIM=128
ADV_LAMBDA=0.5

while getopts "i:w:g:d:e:b:l:r:f:p:s:t:x:y:z:" opt; do
  case $opt in
    i) INPUT_DIM=$OPTARG ;;
    w) NUM_NODES_WM=$OPTARG ;;
    g) NUM_NODES_GM=$OPTARG ;;
    d) DATAPATH=$OPTARG ;;
    e) EPOCHS=$OPTARG ;;
    b) BATCH_SIZE=$OPTARG ;;
    l) LR=$OPTARG ;;
    r) REPETITIONS=$OPTARG ;;
    f) FOLDS=$OPTARG ;;
    p) RESULTPATH=$OPTARG ;;
    s) SEED=$OPTARG ;;
    t) WEIGHT_DECAY=$OPTARG ;;
    x) LAYER=$OPTARG ;;
    y) DROPOUT=$OPTARG ;;
    z) HIDDEN_DIM=$OPTARG ;;
    a) ADV_LAMBDA=$OPTARG ;;
    *) echo "Invalid option: -$OPTARG" ;;
  esac
done

echo "Running with the following parameters:"
echo "Input Dimension: $INPUT_DIM"
echo "Number of Nodes (WM): $NUM_NODES_WM"
echo "Number of Nodes (GM): $NUM_NODES_GM"
echo "Dataset Path: $DATAPATH"
echo "Device: $DEVICE"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Repetitions: $REPETITIONS"
echo "Folds: $FOLDS"
echo "Result Path: $RESULTPATH"
echo "Save Path: $SAVE_PATH"
echo "Seed: $SEED"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Number of Layers: $LAYER"
echo "Dropout: $DROPOUT"
echo "Hidden Dimension: $HIDDEN_DIM"
echo "Adversarial Lambda: $ADV_LAMBDA"

nohup python -u main.py \
  --input_dim=$INPUT_DIM \
  --num_nodes_wm=$NUM_NODES_WM \
  --num_nodes_gm=$NUM_NODES_GM \
  --datapath=$DATAPATH \
  --device=$DEVICE \
  --epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE \
  --lr=$LR \
  --repetitions=$REPETITIONS \
  --folds=$FOLDS \
  --resultpath=$RESULTPATH \
  --savepath=$SAVE_PATH \
  --seed=$SEED \
  --weight_decay=$WEIGHT_DECAY \
  --layer=$LAYER \
  --dropout=$DROPOUT  \
  --hidden_dim=$HIDDEN_DIM \
  --adv_lambda=$ADV_LAMBDA \
  > HCP_result_roi.txt 2>&1 &
