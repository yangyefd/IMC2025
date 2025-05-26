
cd train_cnn
python run_training.py train

python train.py --csv_file ../train_LR/results/combined_model/combined_data.csv \
                --image_dir ../images \
                --output_dir ./cnn_model_output \
                --batch_size 8 \
                --epochs 50 \
                --learning_rate 0.001 \
                --target_size 800 \
                --use_focal_loss

python run_training.py eval

from train_cnn.inference import filter_matches_with_cnn

filtered_matches = filter_matches_with_cnn(
    model_path="./cnn_model_output/best_model.pth",
    matches_dict=original_matches_dict,
    features_data=features_data,
    threshold=0.5,
    image_base_dir="../images"
)