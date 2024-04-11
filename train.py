import torch.nn as nn
import torch, os, datetime, json, gc
from data_loading import Dataset_loading
import config
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from utils import load_checkpoint, save_checkpoint, plot_loss, torch_gc
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model import Discriminator
from model import generator

def train_fn(disc_B, disc_A, gen_A, gen_B, loader, 
             opt_disc:torch.optim.Adam, opt_gen:torch.optim.Adam, 
             l1, mse, d_scaler:torch.cuda.amp.GradScaler, g_scaler:torch.cuda.amp.GradScaler, epoch, loss_dict_avg:dict, max_lenth):
    now = datetime.datetime.now()

    # 将当前时间转换为字符串
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"epoch:{epoch}")
    loop = tqdm(loader, disable=True)
    loss_dict = {"discriminator_A_loss":[], "discriminator_B_loss":[], "generator_A_loss":[], "generator_B_loss":[], "cycle_B_loss":[],"cycle_A_loss":[],"identity_A_loss":[],"identity_B_loss":[], "disc_loss":[], "g_loss":[]}
    folder_path_true = f"./data/save/true/"
    folder_path_false = f"./data/save/fake/"

    
    folder_path_true_use = os.path.join(folder_path_true, str(epoch))
    folder_path_false_use = os.path.join(folder_path_false, str(epoch))
    if not os.path.exists(folder_path_true_use):
        os.makedirs(folder_path_true_use)
    else:
        folder_path_true_other = os.path.join(folder_path_true, str(max_lenth + epoch))
        if not os.path.exists(folder_path_true_other):
            os.makedirs(folder_path_true_other)
            folder_path_true_use = folder_path_true_other
        else:
            folder_path_true_other = os.path.join(folder_path_true, str(max_lenth + epoch) + ' '+ now_str)
            os.makedirs(folder_path_true_other)
            folder_path_true_use = folder_path_true_other

    if not os.path.exists(folder_path_false_use):
        os.makedirs(folder_path_false_use)
    else:
        folder_path_false_other = os.path.join(folder_path_false, str(max_lenth + epoch))
        if not os.path.exists(folder_path_false_other):
            os.makedirs(folder_path_false_other)
            folder_path_false_use = folder_path_false_other
        else:
            folder_path_false_other = os.path.join(folder_path_false, str(max_lenth + epoch) + ' '+ now_str)
            os.makedirs(folder_path_false_other)
            folder_path_false_use = folder_path_false_other
    for idx, (A_image, B_image) in enumerate(loop):
        A_image = A_image.to(config.DEVICE)
        B_image = B_image.to(config.DEVICE) 
        # disc_B and disc_A
        with torch.cuda.amp.autocast():
            fake_B = gen_B(A_image)
            disc_B_real = disc_B(B_image)
            disc_B_fake = disc_B(fake_B.detach())
            disc_B_real_loss = mse(disc_B_real, torch.ones_like(disc_B_real))
            disc_B_fake_loss = mse(disc_B_fake, torch.zeros_like(disc_B_fake))
            disc_B_loss = disc_B_real_loss + disc_B_fake_loss

            fake_A = gen_A(B_image)
            disc_A_real = disc_A(A_image)
            disc_A_fake = disc_A(fake_A.detach())
            disc_A_real_loss = mse(disc_A_real, torch.ones_like(disc_A_real))
            disc_A_fake_loss = mse(disc_A_fake, torch.zeros_like(disc_A_fake))
            disc_A_loss = disc_A_real_loss + disc_A_fake_loss

            

            disc_loss = (disc_B_loss + disc_A_loss) / 2
            loss_dict["discriminator_A_loss"].append(disc_A_loss.cpu().item())
            loss_dict["discriminator_B_loss"].append(disc_B_loss.cpu().item())
            loss_dict["disc_loss"].append(disc_loss.cpu().item())
        opt_disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        with torch.cuda.amp.autocast():
        #generator
        #adveisial loss
            d_A_fake = disc_A(fake_A)
            d_B_fake = disc_B(fake_B)
            gen_A_loss = mse(d_A_fake, torch.ones_like(d_A_fake))
            gen_B_loss = mse(d_B_fake, torch.ones_like(d_B_fake))

            #cycle loss
            cycle_B = gen_B(fake_A)
            cycle_A = gen_A(fake_B)
            cycle_B_loss = l1(B_image, cycle_B)
            cycle_A_loss = l1(A_image, cycle_A)

            #identity_loss
            identity_B = gen_B(B_image)
            identity_A = gen_A(A_image)
            identity_A_loss = l1(B_image, identity_B)
            identity_B_loss = l1(A_image, identity_A)

            g_loss = (
                gen_A_loss + gen_B_loss +
                cycle_B_loss * config.LAMBDA_CYCLE +
                cycle_A_loss * config.LAMBDA_CYCLE +
                identity_B_loss * config.LAMBDA_IDENTITY +
                identity_A_loss * config.LAMBDA_IDENTITY

            )
            loss_dict["generator_A_loss"].append(gen_A_loss.cpu().item())
            loss_dict["generator_B_loss"].append(gen_B_loss.cpu().item())
            loss_dict["cycle_B_loss"].append(cycle_B_loss.cpu().item())
            loss_dict["cycle_A_loss"].append(cycle_A_loss.cpu().item())
            loss_dict["identity_B_loss"].append(identity_B_loss.cpu().item())
            loss_dict["identity_A_loss"].append(identity_A_loss.cpu().item())
            loss_dict["g_loss"].append(g_loss.cpu().item())
        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        if idx % 200 == 0:
            save_image(B_image*0.5 + 0.5, f"{folder_path_true_use}/trueB_{idx}.png")
            save_image(A_image*0.5 + 0.5, f"{folder_path_true_use}/trueA_{idx}.png")
            save_image(fake_B*0.5 + 0.5, f"{folder_path_false_use}/fakeB_{idx}.png")
            save_image(fake_A*0.5 + 0.5, f"{folder_path_false_use}/fakeA_{idx}.png")
        del B_image, A_image, fake_B, fake_A
    plot_loss(loss_dict=loss_dict, i=epoch)
    for key in loss_dict_avg.keys():
        new_value = np.mean(loss_dict.get(key))
        loss_dict_avg[key].append(new_value)
        del new_value
    del loss_dict
    gc.collect()
    return loss_dict_avg




def main():
    file_name = "loss_dict_avg.json"
    # 写入 JSON 文件
    with open(file_name, 'r') as json_file:
        loss_dict_avg = json.load(json_file)
    
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    gen_A = generator(img_channels=3).to(config.DEVICE)
    gen_B = generator(img_channels=3).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_B.parameters()) + list(disc_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    try:
        folders = [int(f) for f in os.listdir(f"./data/save/fake/") if os.path.isdir(os.path.join(f"./data/save/fake/", f))]
        max_lenth = max(folders)
    except Exception as e:
        max_lenth = config.NUM_EPOCHS
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_DISC_B, disc_B, opt_disc, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_DISC_A, disc_A, opt_disc, config.LEARNING_RATE)
    dataset = Dataset_loading(config.TRAIN_DIR_A, config.TRAIN_DIR_B, transform=config.transforms)
    loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    try:
        for epoch in range(config.NUM_EPOCHS):
            loss_dict_avg = train_fn(disc_B, disc_A, gen_A, gen_B, loader, 
                opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch + 1, loss_dict_avg, max_lenth)
            if config.SAVE_MODEL:
                save_checkpoint(gen_B, opt_gen, config.CHECKPOINT_GEN_B)
                save_checkpoint(gen_A, opt_gen, config.CHECKPOINT_GEN_A)
                save_checkpoint(disc_B, opt_disc, config.CHECKPOINT_DISC_B)
                save_checkpoint(disc_A, opt_disc, config.CHECKPOINT_DISC_A)
            torch_gc()
        with open(file_name, 'w') as json_file:
            json.dump(loss_dict_avg, json_file)
        plot_loss(loss_dict_avg, i='avg')
    except Exception as e:
        with open(file_name, 'w') as json_file:
            json.dump(loss_dict_avg, json_file)
        plot_loss(loss_dict_avg, i='avg')
if __name__ == "__main__":
    main()