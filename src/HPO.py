import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import Data 
import Model 
import Loss
import utils
import numpy as np
import os
import joblib

class HPO:

    # Fixed config
    num_classes = 8  
    window_size = 40
    num_future_samples = 10
    input_dim = 19 + num_classes # Number of features per time step
    num_epochs = 50 # early stopping is applied
    early_stopping_patience = 1
    learning_rate = 0.00001 # scheduler is applied
    lr_scheduler_patience = 1

    def __init__(self, train_traces, y_train_traces, val_traces, y_val_traces, isBatchOrganization=True, isReplacementExist=True, isClassWeightsAdjustment=True):
        self.device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
        self.train_traces = train_traces
        self.y_train_traces = y_train_traces
        self.val_traces = val_traces
        self.y_val_traces = y_val_traces
        self.isBatchOrganization = isBatchOrganization
        self.isReplacementExist = isReplacementExist
        self.isClassWeightsAdjustment =  isClassWeightsAdjustment

    def prepare_dataset(self, traces, y_traces, stride, batch_size, isBatchOrganization, isReplacementExist):
        windows = utils.create_sliding_windows(traces, self.window_size, stride)
        windows_labels = utils.generate_labels_for_traces(traces, y_traces, self.window_size, stride)
        dataset = Data.DatasetCreator(windows, windows_labels, self.num_classes, self.num_future_samples, stride)
        if isBatchOrganization:
            if isReplacementExist:
                sampler = Data.BalancedBatchSampler(dataset, batch_size, minority_labels=[5], replacement=True)
            else:
                sampler = Data.BalancedBatchSampler(dataset, batch_size, minority_labels=[5], replacement=False)
            dataloader = DataLoader(dataset, batch_sampler=sampler)
        else:
            dataloader = DataLoader(dataset, batch_size, shuffle=True)
        
        class_weights = utils.calculate_scaled_class_weights(windows_labels, self.num_classes)

        return dataloader, class_weights

    def train(self):
        with wandb.init(project="TS-Anomaly-Detection", config=wandb.config, settings=wandb.Settings(
        _disable_meta=True,
        _disable_git=True,
        _disable_stats=True
        ), anonymous="must" ) as run:
            config = wandb.config

            # Validation: embedding must be divisible by heads
            assert config.embedding_size % config.num_heads == 0, "embedding_size must be divisible by num_heads"

            # DataLoader
            train_dataloader, train_class_weights = self.prepare_dataset(self.train_traces, self.y_train_traces, config.dataset_stride, config.batch_size, self.isBatchOrganization, self.isReplacementExist)
            val_dataloader, val_class_weights = self.prepare_dataset(self.val_traces, self.y_val_traces, config.dataset_stride, config.batch_size, self.isBatchOrganization, self.isReplacementExist)
        
            # Model
            model = Model.Predictor(
                input_dim=self.input_dim,
                embedding_size=config.embedding_size,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                num_classes=self.num_classes,
                window_size=self.window_size,
                num_future_samples=self.num_future_samples,
                dropout=config.dropout
            )

            # Loss and optimizer
            if self.isClassWeightsAdjustment:
                criterion = Loss.Criterion(self.device, train_class_weights)
            else:
                criterion = Loss.Criterion(self.device)

            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.lr_scheduler_patience)

            # Train
            train_losses, val_losses = model.train_model(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                num_epochs=self.num_epochs,
                device=self.device,
                patience=self.early_stopping_patience
            )

            # Log final val loss
            wandb.log({"final_val_loss": val_losses[-1], 'final_train_loss': train_losses[-1]})

            print(f"W&B run {run.name} finished successfully.")
            

