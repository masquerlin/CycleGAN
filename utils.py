import random, torch, os, numpy as np
import torch.nn as nn
import config
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print(checkpoint["optimizer"])
    # print(optimizer.param_groups)
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def plot_loss(loss_dict: dict, i:int):

    num_plots = len(loss_dict)
    cols = 3  # Assuming 2 columns of subplots, you can change this as needed
    rows = -(-num_plots // cols)  # Ceil division to ensure enough rows for all plots
    colors = cm.get_cmap('tab10', num_plots)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    # Flatten axes if needed (when only one row)
    if rows == 1:
        axes = [axes]
    
    for ax, (key, value), color in zip(axes.flat, loss_dict.items(), colors.colors):
        # Convert to numpy array
        value = [x.cpu().detach().numpy() for x in value]
        
        # Plot
        ax.plot(value, label=key, alpha=0.7, color=color)
        ax.set_title(key)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    plt.savefig(f"./model/loss/loss_plot_{i}.png")
