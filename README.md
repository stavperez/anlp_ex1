# Advanced NLP Exercise 1: Fine Tuning

This is the code base for ANLP HUJI course exercise 1, fine tuning pretrained models to perform sentiment analysis on the SST2 dataset.

# Install
``` pip install -r requirements.txt ```

# Fine-Tune and Predict on Test Set
Run:

``` python ex1.py --max_train_samples <number of train samples> --max_eval_samples <number of validation samples> --max_predict_samples <number of prediction samples> --lr <learning rate> --num_train_epochs <number of training epochs> --batch_size <batch size> --do_train/--do_predict --model_path <path to prediction model>```

If you use --do_predict, a prediction.txt file will be generated, containing prediction results for all test samples.

# Running Example:

import os
import shutil

EPOCHS_LIST = [3, 4]
LR_LIST = [5e-5, 3e-5]
BATCH_LIST = [16, 32]

for epochs in EPOCHS_LIST:
    for lr in LR_LIST:
        for batch in BATCH_LIST:
            print(f"=== Training: epochs={epochs}, lr={lr}, batch_size={batch} ===")
            
            # Step 1: Train
            os.system(f"""
            python ex1.py \
                --do_train \
                --max_train_samples -1 \
                --max_eval_samples -1 \
                --num_train_epochs {epochs} \
                --lr {lr} \
                --batch_size {batch}
            """)

            # Step 2: Rename the saved model folder
            model_dir = f"saved_model_e{epochs}_lr{lr}_bs{batch}"
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            shutil.move("saved_model", model_dir)

            # Step 3: Predict using saved model and log test accuracy
            print(f"--- Predicting with model {model_dir} ---")
            os.system(f"""
            python ex1.py \
                --do_predict \
                --max_predict_samples -1 \
                --num_train_epochs {epochs} \
                --lr {lr} \
                --batch_size {batch} \
                --model_path {model_dir}
            """)
