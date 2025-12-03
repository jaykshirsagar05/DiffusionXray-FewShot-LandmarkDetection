import os
import numpy as np
from timeit import default_timer as timer
from tqdm.auto import tqdm 
import torch
import metrics
from torch import nn
import utilities
import csv
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import KFold
## -----------------------------------------------------------------------------------------------------------------##
##                                               TRAINING with GRADIENT ACCUMULATION                                                      ##
## -----------------------------------------------------------------------------------------------------------------##

def train_step(model: torch.nn.Module,
               device: torch.device,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               useHeatmaps: bool = False,
               gradient_accumulation_steps: int = 1):
    # Put model in train mode
    model = model.to(device)
    model.train()

    # Setup train loss value
    train_loss = 0.0
    
    # Loop through data loader data batches
    for batch, data in enumerate(dataloader):

        img_name = data['name']
        images_tensor = data['image']
        landmarks_tensor = data['landmarks']
        heatmaps_tensor = data['heatmaps']

        # Send data to target device
        X = images_tensor.to(device)

        if useHeatmaps:
            y = heatmaps_tensor.to(device)
        else:
            y = landmarks_tensor.to(device)
        

        # print(f"Batch {batch} - image tensor:  {X.shape} - GT tensor: {y.shape}")

        # Forward pass
        y_pred = model(X)

        # print(f"y pred shape: {y_pred.shape} - y shape: {y.shape}")
        
        # Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)

        # normalize loss to account for batch accumulation
        loss = loss / gradient_accumulation_steps
        train_loss += loss.item() * gradient_accumulation_steps # Correcting loss reporting scale

        # Loss backward
        loss.backward()
        
        # Check if it is time to update the weights
        if ((batch + 1) % gradient_accumulation_steps == 0) or (batch + 1 == len(dataloader)):
            # Optimizer step
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()

        # Print progress every 10 batches
        if (batch + 1) % 10 == 0:
            print(f"Batch {batch + 1}/{len(dataloader)} | Loss: {loss.item() * gradient_accumulation_steps:.6f}")

    # Adjust metrics to get average loss and accuracy per batch
    train_loss /= len(dataloader)
    
    return train_loss

## -----------------------------------------------------------------------------------------------------------------##
##                                               VALIDATION PART                                                    ##
## -----------------------------------------------------------------------------------------------------------------##

def validate_step(model: torch.nn.Module,
                  device: torch.device,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  useHeatmaps: bool = False):
    # Put model in eval mode
    model = model.to(device)
    model.eval()

    # Setup validation loss value
    val_loss = 0.0

    with torch.no_grad():
        # Loop through DataLoader batches
        for batch, data in enumerate(dataloader):
            images_tensor = data['image']
            landmarks_tensor = data['landmarks']
            heatmaps_tensor = data['heatmaps']

            # Send data to target device
            X = images_tensor.to(device)

            if useHeatmaps:
                y = heatmaps_tensor.to(device)
            else:
                y = landmarks_tensor.to(device)

            # Forward pass
            val_pred_logits = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Print progress every 10 batches
            if (batch + 1) % 10 == 0:
                print(f"Validation Batch {batch + 1}/{len(dataloader)}")

    # Adjust metrics to get average loss per batch
    val_loss = val_loss / len(dataloader)

    return val_loss


## -----------------------------------------------------------------------------------------------------------------##
##                                               EARLY STOPPING                                                     ##
## -----------------------------------------------------------------------------------------------------------------##

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0, save_path=not None, counter=0, best_val_loss=None):
        self.patience = patience
        self.counter = counter
        self.best_val_loss = best_val_loss
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = save_path

    def call(self, val_loss, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, epoch):

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            save_model(self.path, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, self.best_val_loss, epoch, called_by_early_stopping=True)

        elif val_loss >= self.best_val_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = val_loss
            save_model(self.path, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, self.best_val_loss, epoch, called_by_early_stopping=True)
            self.counter = 0


## -----------------------------------------------------------------------------------------------------------------##
##                                           SAVE AND LOAD A MODEL                                            ##
## -----------------------------------------------------------------------------------------------------------------##
def save_model(save_path, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, best_val_loss, epoch, called_by_early_stopping=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if called_by_early_stopping:
        checkpoint_path = os.path.join(save_path, "best_checkpoint.pt")
    else:
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch{epoch}.pt")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_fn': loss_fn.state_dict(),
        'results': results,
        'epochs_without_improvement': epochs_without_improvement,
        'best_val_loss': best_val_loss,
        'epoch': epoch
    }, checkpoint_path)
    #print(f"Model saved to {checkpoint_path}")

    
def load_model(load_path, model, optimizer, scheduler, loss_fn, device):
    checkpoint = torch.load(load_path, map_location=torch.device(device))

    # Load the state_dict into the model only if it exists in the checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)  # Move the model to the specified device

    # Load the optimizer state_dict only if it exists in the checkpoint
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load the scheduler state_dict only if it exists in the checkpoint
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load the loss_fn state_dict only if it exists in the checkpoint
    if 'loss_fn' in checkpoint:
        loss_fn.load_state_dict(checkpoint['loss_fn'])

    # Load other values only if they exist in the checkpoint
    start_epoch = checkpoint.get('epoch', 0) + 1
    results = checkpoint.get('results', None)
    epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
    best_val_loss = checkpoint.get('best_val_loss', None)
    print(f"Model loaded from {load_path} | Starting from epoch {start_epoch} | Best validation loss: {best_val_loss} | Epochs without improvement: {epochs_without_improvement}")
    return model, optimizer, scheduler, loss_fn, start_epoch, results, epochs_without_improvement, best_val_loss


## -----------------------------------------------------------------------------------------------------------------##
##                                           TRAINING + VALIDATION PART                                             ##
## -----------------------------------------------------------------------------------------------------------------##

def train_and_validate(model: torch.nn.Module,
                       device: torch.device,
                       train_dataloader: torch.utils.data.DataLoader,
                       val_dataloader: torch.utils.data.DataLoader,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler,
                       loss_fn: torch.nn.Module,
                       epochs: int = 10,
                       save_path: str = None,
                       useHeatmaps: bool = True,
                       patience: int = 10,
                       save_all_epochs: bool = False,
                       useGradAcc: int = 1,
                       continue_training: bool = False):
    
    if continue_training:
        model_path = os.path.join(save_path, "best_checkpoint.pt")
        assert model_path is not None, "If you want to continue training, you must provide a path to load the model from."

        # Load the model from the path
        model, optimizer, scheduler, loss_fn, start_epoch, results, epochs_without_improvement, best_val_loss = load_model(model_path, model, optimizer, scheduler, loss_fn, device)
    else:
        # Create empty results dictionary and initialize epoch
        results = {"train_loss": [], "val_loss": []}
        start_epoch = 1
        best_val_loss = float("inf")
        epochs_without_improvement = 0

    # Start the timer
    start_time = timer()

    # Create EarlyStopping instance
    early_stopping = EarlyStopping(patience=patience, save_path=save_path, counter=epochs_without_improvement, best_val_loss=best_val_loss)

    # Track LR per epoch
    lrs_per_epoch = []

    # Loop through training and validating steps for a number of epochs
    for epoch in tqdm(range(start_epoch, epochs + 1)):
        print("entered epoch loop DEBPUGG")
        assert useGradAcc >= 1, "Gradient accumulation steps must be greater than 1"

        train_loss = train_step(model, device, train_dataloader, loss_fn, optimizer, useHeatmaps, gradient_accumulation_steps=useGradAcc)
        val_loss = validate_step(model, device, val_dataloader, loss_fn, useHeatmaps)

        scheduler_type = scheduler.__class__.__name__
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Capture current LR (first param group)
        current_lr = optimizer.param_groups[0]["lr"]
        lrs_per_epoch.append(current_lr)

        # Print out what's happening
        print(f"Epoch {epoch} | Train Loss: {train_loss:.7f} | Validation Loss: {val_loss:.7f} | LR: {current_lr:.6e}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        # Save the trained model
        if save_all_epochs is True:
            save_model(save_path, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, best_val_loss, epoch)

        # Check for early stopping
        early_stopping.call(val_loss, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    # Persist curves
    try:
        _save_training_curves(results, lrs_per_epoch, save_path or "./")
    except Exception as e:
        print(f"Warning: failed to save training curves: {e}")

    # Return the filled results at the end of the epochs
    return results

## -----------------------------------------------------------------------------------------------------------------##
##                                                  FINE-TUNING IN-DOMAIN                                           ##
## -----------------------------------------------------------------------------------------------------------------##
def fine_tune(model: torch.nn.Module,
              device: torch.device,
              train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler,
              loss_fn: torch.nn.Module,
              epochs: int = 10,
              load_path: str = None,
              save_path: str = None,
              useHeatmaps: bool = True,
              patience: int = 10,
              useGradAcc: int = 1):
    
    assert load_path is not None, "You must provide a path to load the model from."
    
    # Load the model from the path
    model.load_state_dict(torch.load(load_path, map_location=torch.device(device)), strict=False) 
    model = model.to(device)  # Move the model to the specified device
    
    # Create empty results dictionary and initialize epoch
    results = {"train_loss": [], "val_loss": []}
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    
    # Start the timer
    start_time = timer()

    # Create EarlyStopping instance
    early_stopping = EarlyStopping(patience=patience, save_path=save_path, counter=epochs_without_improvement, best_val_loss=best_val_loss)

    # Loop through training and validating steps for a number of epochs
    for epoch in tqdm(range(start_epoch, epochs + 1)):

        assert useGradAcc >= 1, "Gradient accumulation steps must be greater than 1"

        train_loss = train_step(model, device, train_dataloader, loss_fn, optimizer, useHeatmaps, gradient_accumulation_steps=useGradAcc)

        val_loss = validate_step(model, device, val_dataloader, loss_fn, useHeatmaps)

        scheduler_type = scheduler.__class__.__name__
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            # Update the learning rate using the scheduler
            scheduler.step()

        # Print out what's happening
        print(f"Epoch {epoch} | Train Loss: {train_loss:.7f} | Validation Loss: {val_loss:.7f}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        # Check for early stopping
        early_stopping.call(val_loss, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    # Return the filled results at the end of the epochs
    return results   
                 


## -----------------------------------------------------------------------------------------------------------------##
##                                                  EVALUATION PART                                                 ##
## -----------------------------------------------------------------------------------------------------------------##
def test_step(model: torch.nn.Module,
              device: torch.device,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              num_landmarks: int,
              useHeatmaps: bool = False,
              sigma: int = 1.5,
              load_path: str = None):
    # Take the baseline of the path
    if load_path is not None:
        model_dir = os.path.dirname(load_path)
        
    # Put model in eval mode
    model = model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    model_encoder = model.encoder.__class__.__name__ if hasattr(model, 'encoder') else ""

    # Setup test loss and test accuracy values
    test_loss = 0.0
    results = {}
    distances = [] 

    with torch.no_grad():
        # Loop through DataLoader batches
        for batch, data in enumerate(dataloader):
            images_name = data['name']
            images_tensor = data['image']
            landmarks_tensor = data['landmarks']
            heatmaps_tensor = data['heatmaps']
            original_size = data['original_size']
            resized_size = data['resized_size']
            spacing_tensor = data['spacing'] if 'spacing' in data else None  # LUNA16 provides spacing
            # NEW: number of valid landmarks per sample
            num_valid_batch = data['num_valid_landmarks'] if 'num_valid_landmarks' in data else None

            # Send data to target device
            X = images_tensor.to(device)

            if useHeatmaps:
                y = heatmaps_tensor.to(device)
            else:
                y = landmarks_tensor.to(device)

            # Forward pass
            y_pred = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Move prediction to CPU numpy
            y_pred = y_pred.cpu()

            # Decide if GT heatmaps are fused
            fused_flag = False
            if useHeatmaps:
                # Shapes:
                # 2D non‑fused: (B, N, H, W)
                # 2D fused:     (B, 1, H, W) or (B, H, W)
                # 3D non‑fused: (B, N, D, H, W)
                # 3D fused:     (B, 1, D, H, W) or (B, D, H, W)
                gt_shape = tuple(heatmaps_tensor.shape)
                if heatmaps_tensor.ndim >= 3:
                    if heatmaps_tensor.ndim in (4, 5):
                        channels_dim = 1
                        if gt_shape[channels_dim] != num_landmarks:
                            fused_flag = True
                    elif heatmaps_tensor.ndim in (3,):
                        fused_flag = True
            else:
                fused_flag = False  # keypoint-regression path should compute keypoint mAP

            # Compute metrics
            mse_list, mAP_list_heatmaps, mAP_list_keypoints, iou_list, distance_list = metrics.compute_batch_metrics(
                landmarks_tensor,
                heatmaps_tensor,
                y_pred,
                resized_size,
                num_landmarks,
                useHeatmaps,
                sigma,
                spacing_batch=spacing_tensor,
                fused=fused_flag,
                num_valid_batch=num_valid_batch  # NEW: filter padded landmarks
            )
            # Append to full list for MRE/SDR
            distances.extend(distance_list)
            
            # Store image names as keys and their corresponding predictions as values.
            for i, name in enumerate(images_name):
                # Keypoint mAP may be undefined when fused; guard it
                map2_val = mAP_list_keypoints[i] if (not fused_flag and i < len(mAP_list_keypoints)) else float('nan')
                results[name] = { 
                    'prediction': y_pred[i],
                    'mse': mse_list[i],
                    'map1': mAP_list_heatmaps[i],
                    'map2': map2_val,
                    'iou': iou_list[i]
                }

            del batch, data, images_name, images_tensor, landmarks_tensor, heatmaps_tensor, original_size, resized_size, X, y, y_pred, loss, mse_list, mAP_list_heatmaps, mAP_list_keypoints, iou_list, distance_list   # Free memory
                                       

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)

    # Compute metrics on full list
    mre = metrics.compute_mre(distances)
    sdr = metrics.compute_sdr(distances)

    return test_loss, results, mre, sdr


def evaluate(model: torch.nn.Module,
          device: torch.device,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          load_path: str,
          num_landmarks: int = 6,
          useHeatmaps: bool = True,
          sigma: int = 1.5,
          currentKfold: int = 1,
          res_file_path: str = "results/readable_res.csv"):
    
    checkpoint = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nModel loaded from {load_path}")
    epoch = checkpoint.get('epoch', "Undefined")

    # Get the loss and the predictions dictionary
    test_loss, results, mre, sdr = test_step(model, device, test_dataloader, loss_fn, num_landmarks, useHeatmaps, sigma, load_path)

    # Determine experiment dir to drop artifacts
    exp_dir = os.path.dirname(load_path) if load_path else "./"
    try:
        _save_testing_curves(results, mre, sdr, exp_dir, prefix="validation" if "val" in exp_dir.lower() else "test")
    except Exception as e:
        print(f"Warning: failed to save testing curves: {e}")

    total_mse_list = []
    total_mAP_heatmaps_list = []
    total_mAP_keypoints_list = []
    total_iou_list = []

    # Collect metrics of all images
    for value in results.values():
        total_mse_list.append(value['mse'])
        total_mAP_heatmaps_list.append(value['map1'])
        total_mAP_keypoints_list.append(value['map2'])
        total_iou_list.append(value['iou'])

    # Means (ignore NaNs for keypoint mAP when fused)
    total_mse_mean = np.mean(total_mse_list)
    total_mAP_heatmaps_mean = np.mean(total_mAP_heatmaps_list)
    total_mAP_keypoints_mean = np.nanmean(total_mAP_keypoints_list)
    total_iou_mean = np.mean(total_iou_list)

    # STDs
    total_mse_std = np.std(total_mse_list)
    total_mAP_heatmaps_std = np.std(total_mAP_heatmaps_list)
    total_mAP_keypoints_std = np.nanstd(total_mAP_keypoints_list)
    total_iou_std = np.std(total_iou_list)

    # Create a string representation of the sdr dictionary
    sdr_str = '\n'.join(f'\tThresholds {k}: {v*100:.2f}' for k, v in sorted(sdr.items()))

    # Print and Save results
    res_file = open(res_file_path, 'a')
    print(f"\n{load_path}", file=res_file)
    print(f"Fold {currentKfold} - Epoch: {epoch} | MSE: {total_mse_mean:.2f} ± {total_mse_std:.2f} | mAP heat: {total_mAP_heatmaps_mean:.2f} ± {total_mAP_heatmaps_std:.2f} | mAP key: {total_mAP_keypoints_mean:.2f} ± {total_mAP_keypoints_std:.2f} | IoU: {total_iou_mean:.2f} ± {total_iou_std:.2f} \nMRE: {mre:.2f} \nSDR: \n{sdr_str}", file=res_file)
    res_file.close()

    print(f"Fold {currentKfold} - Epoch: {epoch} | \nMSE: {total_mse_mean:.2f} ± {total_mse_std:.2f} | \nmAP heat: {total_mAP_heatmaps_mean:.2f} ± {total_mAP_heatmaps_std:.2f} | mAP key: {total_mAP_keypoints_mean:.2f} ± {total_mAP_keypoints_std:.2f} | \nIoU: {total_iou_mean:.2f} ± {total_iou_std:.2f} | \nMRE: {mre:.2f} | \nSDR: \n{sdr_str}")
    del total_mse_list, total_mAP_heatmaps_list, total_mAP_keypoints_list, total_iou_list

    return test_loss, results, mre, sdr, total_mse_mean, total_mAP_heatmaps_mean, total_mAP_keypoints_mean, total_iou_mean, epoch




# ------------------------------------------------------------------------
#                               Reinstantiate Model
# ------------------------------------------------------------------------

def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)
     
def reinstantiate_model(model, optimizer, scheduler):
    model_type = model.__class__.__name__
    scheduler_type = scheduler.__class__.__name__
    optimizer_type = optimizer.__class__.__name__
    #print(scheduler_params)
    
    reset_all_weights(model)

    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=optimizer.param_groups[0]['lr'])
    else:    
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler.factor, patience=scheduler.patience, verbose=True, mode=scheduler.mode)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return model, optimizer, scheduler


## -----------------------------------------------------------------------------------------------------------------##
##                                                            K-FOLD                                             ##
## -----------------------------------------------------------------------------------------------------------------##

        
def k_fold_train_and_validate(model: torch.nn.Module,
                                device: torch.device,
                                train_dataset: torch.utils.data.Dataset,
                                optimizer: torch.optim.Optimizer,
                                scheduler: torch.optim.lr_scheduler,
                                loss_fn: torch.nn.Module,
                                epochs: int,
                                early_stopping: int,
                                batch_size: int,
                                gradient_accumulation_steps: int,
                                num_landmarks: int,
                                sigma: int,
                                save_model_path: str,
                                log_file: str,
                                k_folds: int = 5,
                                onlyInference: bool = True
                                ):
    
    if onlyInference:
        k_train_losses = [0]
        k_val_losses = [0]
    else:
        k_train_losses = []
        k_val_losses = []

    k_test_losses = []
    k_mse = []
    k_iou = []
    k_map_heat = []
    k_map_key = []
    k_mre = []
    k_sdr = {}

    results_folds = []
        
    # Get the total number of samples
    total_size = len(train_dataset)

    # Divide by the number of folds to get the size of each fold
    fold_size = total_size // k_folds

    indices = list(range(total_size))
  

    for fold in range(k_folds):
        
        # Assign the fold as the val set
        val_ids = indices[fold*fold_size:(fold+1)*fold_size]

        # The remaining data will be used for training 
        train_ids = indices[:fold*fold_size] + indices[(fold+1)*fold_size:]

        # Create the subsets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Create the data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=4, pin_memory=True)

        save_fold_path = f"{save_model_path}/fold_{fold}"
        print(f"Training fold {fold}...")
        print(f"Path: {save_fold_path}")
        
        if not onlyInference:
            
            model, optimizer, scheduler = reinstantiate_model(model, optimizer, scheduler)
            
            # Train on the current fold
            fold_train_results = train_and_validate(model, device, train_loader, val_loader, optimizer, scheduler, loss_fn, epochs, 
                                            save_fold_path, patience=early_stopping, useGradAcc=gradient_accumulation_steps, continue_training=False)
            
            last_train_loss = fold_train_results['train_loss'][-1]
            last_val_loss = fold_train_results['val_loss'][-1]

            k_train_losses.append(last_train_loss)
            k_val_losses.append(last_val_loss)

            print(f"FOLD {fold} | Train loss: {last_train_loss} | Val loss: {last_val_loss}")
            del fold_train_results, last_train_loss, last_val_loss, train_loader, train_subsampler, val_subsampler, train_ids, val_ids


        # ---------------------- Evaluate performances on val set -------------------------------
        load_fold_path = os.path.join(save_fold_path, f"best_checkpoint.pt")
        # FIX: pass useHeatmaps explicitly and sigma correctly
        test_loss, results, mre, sdr, mse, mAP_heatmaps, mAP_keypoints, iou, epoch = evaluate(
            model, device, val_loader, loss_fn, load_fold_path,
            num_landmarks=num_landmarks, useHeatmaps=True, sigma=sigma, res_file_path=log_file
        )

        k_test_losses.append(test_loss)
        
        k_mre.append(mre)
        
        # Update the sdr dictionary
        for threshold, value in sdr.items():
            if threshold not in k_sdr:
                k_sdr[threshold] = []
            k_sdr[threshold].append(value)

        # Create a list with all metrics of all images
        for value in results.values():
            k_mse.append(value['mse'])
            k_map_heat.append(value['map1'])
            k_map_key.append(value['map2'])
            k_iou.append(value['iou'])    

        del test_loss, results, mre, sdr, load_fold_path, val_loader, 
        
    # Compute the mean and SD for each threshold
    sdr_mean_std = {threshold: (np.mean(values), np.std(values)) for threshold, values in k_sdr.items()}

    # Compute the mean for the losses
    k_train_loss_mean = np.mean(k_train_losses)
    k_train_loss_std = np.std(k_train_losses)

    k_val_loss_mean = np.mean(k_val_losses)
    k_val_loss_std = np.std(k_val_losses)

    k_test_loss_mean = np.mean(k_test_losses)
    k_test_loss_std = np.std(k_test_losses)

    # Compute the mean between all samples
    k_mse_mean = np.mean(k_mse)
    k_map_heat_mean = np.mean(k_map_heat)
    k_map_key_mean = np.mean(k_map_key)
    k_iou_mean = np.mean(k_iou)

    # Compute the standard deviation between all samples
    k_mse_std = np.std(k_mse)
    k_map_heat_std = np.std(k_map_heat)
    k_map_key_std = np.std(k_map_key)
    k_iou_std = np.std(k_iou)

    # Compute the mean MRE and mean SDR
    k_mre_mean = np.mean(k_mre)
    k_mre_std = np.std(k_mre)

    res_file = open(log_file, 'a')
    print(f"----------------------------------------------------------------- GLOBAL RES for {k_folds} Folds \n",
        f"Train loss ---> Mean: {k_train_loss_mean} | Std: {k_train_loss_std} \n",
        f"Val loss ---> Mean: {k_val_loss_mean} | Std: {k_val_loss_std} \n",
        f"Test loss ---> Mean: {k_test_loss_mean} | Std: {k_test_loss_std} \n",
        f"MSE ---> Mean: {k_mse_mean:.2f} | Std: {k_mse_std:.2f} \n",
        f"mAp heat ---> Mean: {k_map_heat_mean:.2f} | Std: {k_map_heat_std:.2f} \n",
        f"mAp key ---> Mean: {k_map_key_mean:.2f} | Std: {k_map_key_std:.2f} \n",
        f"IOU ---> Mean: {k_iou_mean:.2f} | Std: {k_iou_std:.2f} \n",
        f"MRE ---> Mean: {k_mre_mean:.2f} | Std: {k_mre_std:.2f} \n",
        f"SDR:\n",
        *(f"Threshold {threshold}: Mean: {mean*100:.2f} | Std: {std*100:.2f}\n" for threshold, (mean, std) in sdr_mean_std.items()),
        file=res_file)
    res_file.close()


    print(f"----------------------------------------------------------------- GLOBAL RES for {k_folds} Folds \n",
        f"Train loss ---> Mean: {k_train_loss_mean} | Std: {k_train_loss_std} \n",
        f"Val loss ---> Mean: {k_val_loss_mean} | Std: {k_val_loss_std} \n",
        f"Test loss ---> Mean: {k_test_loss_mean} | Std: {k_test_loss_std} \n",
        f"MSE ---> Mean: {k_mse_mean:.2f} | Std: {k_mse_std:.2f} \n",
        f"mAp heat ---> Mean: {k_map_heat_mean:.2f} | Std: {k_map_heat_std:.2f} \n",
        f"mAp key ---> Mean: {k_map_key_mean:.2f} | Std: {k_map_key_std:.2f} \n",
        f"IOU ---> Mean: {k_iou_mean:.2f} | Std: {k_iou_std:.2f} \n",
        f"MRE ---> Mean: {k_mre_mean:.2f} | Std: {k_mre_std:.2f} \n",
        f"SDR:\n",
        *(f"Threshold {threshold}: Mean: {mean*100:.2f} | Std: {std*100:.2f}\n" for threshold, (mean, std) in sdr_mean_std.items()))
    del k_train_losses, k_val_losses, k_test_losses, k_mse, k_iou, k_map_heat, k_map_key, k_mre, k_sdr, results_folds, train_dataset, total_size, fold_size, indices

def _save_training_curves(results, lrs_per_epoch, save_path):
    os.makedirs(save_path, exist_ok=True)
    # Save JSON + CSV
    with open(os.path.join(save_path, "training_results.json"), "w") as f:
        json.dump({"train_loss": results["train_loss"], "val_loss": results["val_loss"], "lr": lrs_per_epoch}, f, indent=2)
    with open(os.path.join(save_path, "training_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
        for i in range(len(results["train_loss"])):
            writer.writerow([i + 1, results["train_loss"][i], results["val_loss"][i], lrs_per_epoch[i] if i < len(lrs_per_epoch) else ""])

    # Plot Loss curves
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(results["train_loss"])+1), results["train_loss"], label="Train")
    plt.plot(range(1, len(results["val_loss"])+1), results["val_loss"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training/Validation Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curves.png"))
    plt.close()

    # Plot LR curve
    if lrs_per_epoch:
        plt.figure(figsize=(8,5))
        plt.plot(range(1, len(lrs_per_epoch)+1), lrs_per_epoch)
        plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.title("Learning Rate per Epoch"); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "lr_curve.png"))
        plt.close()

def _save_testing_curves(results_dict, mre, sdr, save_path, prefix="test"):
    os.makedirs(save_path, exist_ok=True)

    # Flatten lists
    mse = [v['mse'] for v in results_dict.values()]
    map_heat = [v['map1'] for v in results_dict.values()]
    map_key = [v['map2'] for v in results_dict.values() if not (v['map2'] is None or (isinstance(v['map2'], float) and str(v['map2'])=='nan'))]
    iou = [v['iou'] for v in results_dict.values()]

    # Save CSV per-image
    with open(os.path.join(save_path, f"{prefix}_per_image_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mse", "mAP_heat", "mAP_key", "iou"])
        for k, v in results_dict.items():
            writer.writerow([k, v['mse'], v['map1'], v['map2'], v['iou']])

    # Save summary JSON
    summary = {
        "mre": float(mre),
        "sdr": {str(k): float(v) for k, v in sdr.items()},
        "mse_mean": float(np.mean(mse)) if mse else None,
        "mAP_heat_mean": float(np.mean(map_heat)) if map_heat else None,
        "mAP_key_mean": float(np.nanmean(map_key)) if map_key else None,
        "iou_mean": float(np.mean(iou)) if iou else None
    }
    with open(os.path.join(save_path, f"{prefix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Histograms/Boxplots
    def _hist(data, name, bins=20):
        if not data: return
        plt.figure(figsize=(7,4))
        plt.hist(data, bins=bins, alpha=0.8, color="#3b82f6")
        plt.title(f"{name} Histogram"); plt.xlabel(name); plt.ylabel("Count"); plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{prefix}_{name.lower()}_hist.png"))
        plt.close()
    def _box(data, name):
        if not data: return
        plt.figure(figsize=(5,5))
        plt.boxplot(data, vert=True)
        plt.title(f"{name} Boxplot")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{prefix}_{name.lower()}_box.png"))
        plt.close()

    _hist(mse, "MSE"); _box(mse, "MSE")
    _hist(map_heat, "mAP_heat"); _box(map_heat, "mAP_heat")
    _hist(map_key, "mAP_key"); _box(map_key, "mAP_key")
    _hist(iou, "IoU"); _box(iou, "IoU")

    # SDR bar chart
    if sdr:
        thresholds = [str(k) for k in sorted(sdr.keys())]
        values = [sdr[k] for k in sorted(sdr.keys())]
        plt.figure(figsize=(8,5))
        plt.bar(thresholds, np.array(values)*100.0, color="#10b981")
        plt.title("SDR (%) by threshold"); plt.xlabel("Threshold"); plt.ylabel("SDR (%)"); plt.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{prefix}_sdr_bar.png"))
        plt.close()