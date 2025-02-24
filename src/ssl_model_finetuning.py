"""
    python ssl_model_finetuning.py \
    --model facebook/wav2vec2-base \
    --df_data data.csv \
    --output_dir results \
    --batch 16 \
    --epochs 50 \
    --warmup_steps 500 \
    --lr 1e-4 \
    --max_duration 7.0 \
    --seed 42
    --output_file results.txt
"""

import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import EarlyStoppingCallback
import pandas as pd
import argparse
import random
import numpy as np
import os
import librosa

import warnings
warnings.filterwarnings("ignore")

from dataset import Dataset
from utils import WeightedTrainer, define_training_args, compute_metrics, compute_class_weights
    

"""
    Define Command Line Parser 
"""
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="vocal")
    parser.add_argument(
        "--batch",
        help="batch size",
        default=16, 
        type=int,
        required=False)
    parser.add_argument(
        "--epochs",
        help="number of training epochs",
        default=50, 
        type=int,
        required=False)        
    parser.add_argument(
        "--model",
        help="model to use",
        default="facebook/wav2vec2-base",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--df_data",
        help="path to the data df",
        default="data.csv",
        type=str,
        required=False) 
    parser.add_argument(
        "--dataset_root",
        help="path to the dataset root",
        default="vocalization_data",
        type=str,
        required=False) 
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="results",
        type=str,
        required=False)
    parser.add_argument(
        "--output_file",
        help="path to the output file",
        default="results.txt",
        type=str,
        required=False)
    parser.add_argument(
        "--warmup_steps",
        help="number of warmup steps",
        default=500,
        type=int,
        required=False)
    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
        required=False)
    parser.add_argument(
        "--max_duration",
        help="Maximum audio duration",
        default=7.0,
        type=float,
        required=False) 
    parser.add_argument(
        "--seed",
        help="seed",
        default=42,
        type=int,
        required=False)
    args = parser.parse_args()
    return args


""" 
    Read and Process Data
"""
def read_data(df_train, df_val, label_name="emotion"):
    
    ## Prepare Labels
    labels = df_train[label_name].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    ## Train
    for index in range(0,len(df_train)):
        df_train.loc[index,'label'] = label2id[df_train.loc[index,label_name]]
    df_train['label'] = df_train['label'].astype(int)

    ## Validation
    for index in range(0,len(df_val)):
        df_val.loc[index,'label'] = label2id[df_val.loc[index,label_name]]
    df_val['label'] = df_val['label'].astype(int)

    return df_train, df_val, num_labels, label2id, id2label, labels


""" 
    Define model and feature extractor 
"""
def define_model(
    model_checkpoint, 
    num_labels, 
    label2id, 
    id2label, 
    device="cuda"
    ):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    if model_checkpoint.startswith("facebook") or model_checkpoint.startswith("ALM"):
        print("Loading model from HF Hub...")
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
            ).to(device)
    else:
        print("Loading model from local files...")
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            local_files_only=True,
            ignore_mismatched_sizes=True
            ).to(device)
    return feature_extractor, model


""" 
    Main Program 
"""
if __name__ == '__main__':

    ## Utils 
    args = parse_cmd_line_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print("------------------------------------")
    print("Running with the following parameters:")
    print("Batch size: ", args.batch)
    print("Number of epochs: ", args.epochs)
    print("Model: ", args.model)
    print("Warmup steps: ", args.warmup_steps)
    print("Learning rate: ", args.lr)
    print("Maximum audio duration: ", args.max_duration)
    print("------------------------------------\n")

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir) 

    ## Set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ## Load the data
    data = pd.read_csv(args.df_data, index_col=None)

    ## Split the data into 10 folds
    indices = list(range(len(data)))
    random.shuffle(indices)
    fold_size = len(data) // 10
    folds = []
    for i in range(10):
        start = i * fold_size
        end = (i + 1) * fold_size
        if i == 9:
            end = len(data)
        folds.append(indices[start:end])

    uars = []
    f1_macros = []
    accuracies = []

    ## Train the model on each fold
    for fold in range(10):

        try:

            print(f"Fold {fold + 1}")

            ## Split the data into train and validation
            train_indices = []
            for i in range(10):
                if i != fold:
                    train_indices.extend(folds[i])
            val_indices = folds[fold]
            df_train = data.iloc[train_indices].reset_index(drop=True)
            df_val = data.iloc[val_indices].reset_index(drop=True)
            print("Train: ", len(df_train))   
            print("Val: ", len(df_val))

            ## Prepare Labels
            df_train, df_val, num_labels, label2id, id2label, labels = read_data(
                df_train=df_train,
                df_val=df_val,
                label_name="emotion"
                )
            print("Num labels: ", num_labels)

            ## Load the model and feature extractor
            feature_extractor, model = define_model(
                model_checkpoint=args.model, 
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label,
                device=device
                )

            ## Train & Val Datasets 
            max_duration = args.max_duration
            train_dataset = Dataset(df_train, args.dataset_root, feature_extractor, max_duration)
            val_dataset = Dataset(df_val, args.dataset_root, feature_extractor, max_duration)

            ## Training Arguments and Class Weights
            training_arguments = define_training_args(
                output_dir=output_dir, 
                batch_size=args.batch, 
                num_epochs=args.epochs,
                lr=args.lr, 
                gradient_accumulation_steps=1,
                warmup_steps=args.warmup_steps,
                )
            class_weights = compute_class_weights(df_train)

            ## Trainer 
            trainer = WeightedTrainer(
                class_weights=class_weights,
                model=model,
                args=training_arguments,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
                )

            ## Train and Evaluate
            trainer.train()

            ## Evaluate
            print("Evaluating...")
            predictions = trainer.predict(val_dataset)
            accuracies.append(predictions.metrics['test_accuracy'])
            f1_macros.append(predictions.metrics['test_f1_macro'])
            uars.append(predictions.metrics['test_UAR'])

        except Exception as e:
            print("Error: ", e)
            print("Error in fold: ", fold)
            continue

    with open(args.output_file, "a+") as f:
        f.write(f"Dataset: {args.df_data}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"UAR: {np.mean(uars)} ± {np.std(uars)}\n")
        f.write(f"Accuracy: {np.mean(accuracies)} ± {np.std(accuracies)}\n")
        f.write(f"F1 Macro: {np.mean(f1_macros)} ± {np.std(f1_macros)}\n")
        f.write("------------------------------------------------\n\n")
