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
        # Plot
        ax.plot(value, label=key, alpha=0.7, color=color)
        ax.set_title(key)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.legend()
    delete_ax(fig=fig, axes=axes)
    # Adjust layout
    plt.tight_layout()
    
    plt.savefig(f"./model/loss/loss_plot_{i}.png")
    plt.close()
def delete_ax(fig, axes):
        """
        Delete axes from a figure based on whether they contain any plots.

        Args:
            fig (matplotlib.figure.Figure): Matplotlib figure.
            axes (matplotlib.axes.Axes): Axes to check and potentially delete.
        """
        if not isinstance(axes, plt.Axes):
            if len(axes.shape) == 1:
                for i in range(axes.size):
                    ax = axes[i]
                    if not any([len(ax.lines), len(ax.collections), len(ax.patches)]):
                        fig.delaxes(ax)
            else:
                for i in range(axes.shape[0]):
                    for j in range(axes.shape[1]):
                        ax = axes[i,j]
                        if not any([len(ax.lines), len(ax.collections), len(ax.patches)]):
                            fig.delaxes(ax)
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(config.DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()