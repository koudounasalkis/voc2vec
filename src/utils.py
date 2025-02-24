from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils import class_weight


""" 
    Trainer Class 
"""  
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").long()  
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss



""" 
    Define training arguments 
""" 
def define_training_args(
    output_dir, 
    batch_size, 
    num_epochs=50,
    lr=1.0e-4, 
    gradient_accumulation_steps=1, 
    warmup_steps=500,
    ): 
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        logging_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        remove_unused_columns=False
        )
    return training_args


""" 
    Define Class Weights 
"""
def compute_class_weights(df_train):
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(df_train["label"]),
        y=np.array(df_train["label"])
    )
    class_weights = torch.tensor(class_weights, device="cuda", dtype=torch.float32)
    return class_weights


""" 
    Define Metrics
"""
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    uar = recall_score(labels, preds, average='macro')
    
    return { 
        'accuracy': acc, 
        'f1_macro': f1_macro, 
        'UAR': uar
        }