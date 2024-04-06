import torch.nn as nn
import torch
from data_loading import Dataset_loading
import config
import torch.optim as optim
from tqdm import tqdm
from utils import load_checkpoint, save_checkpoint, plot_loss
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model import Discriminator
from model import generator

def train_fn(disc_B, disc_A, gen_A, gen_B, loader, 
             opt_disc:torch.optim.Adam, opt_gen:torch.optim.Adam, 
             l1, mse, d_scaler:torch.cuda.amp.GradScaler, g_scaler:torch.cuda.amp.GradScaler, epoch):
    loop = tqdm(loader)
    loss_dict = {"discriminator_A_loss":[], "discriminator_B_loss":[], "discriminator_both_loss":[], "generator_A_loss":[], "generator_B_loss":[], "cycle_B_loss":[],"cycle_A_loss":[],"identity_A_loss":[],"identity_B_loss":[]}

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

            loss_dict["discriminator_A_loss"].append(disc_A_loss)
            loss_dict["discriminator_B_loss"].append(disc_B_loss)
            loss_dict["discriminator_both_loss"].append(disc_loss)

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
            loss_dict["generator_A_loss"].append(gen_A_loss)
            loss_dict["generator_B_loss"].append(gen_B_loss)
            loss_dict["cycle_B_loss"].append(cycle_B_loss)
            loss_dict["cycle_A_loss"].append(cycle_A_loss)
            loss_dict["identity_B_loss"].append(identity_B_loss)
            loss_dict["identity_A_loss"].append(identity_A_loss)
        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 ==0:
            save_image(B_image*0.5+0.5, f"./data/save/true/trueB_{idx}.png")
            save_image(A_image*0.5+0.5, f"./data/save/true/trueA_{idx}.png")
            save_image(fake_B*0.5+0.5, f"./data/save/fake/fakeB_{idx}.png")
            save_image(fake_A*0.5+0.5, f"./data/save/fake/fakeA_{idx}.png")
    plot_loss(loss_dict=loss_dict, i=epoch)




def main():
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

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_DISC_B, disc_B, opt_disc, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_DISC_A, disc_A, opt_disc, config.LEARNING_RATE)
    dataset = Dataset_loading(config.TRAIN_DIR_A, config.TRAIN_DIR_B, transform=config.transforms)
    loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_B, disc_A, gen_A, gen_B, loader, 
            opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch)
        if config.SAVE_MODEL:
            save_checkpoint(gen_B, opt_gen, config.CHECKPOINT_GEN_B)
            save_checkpoint(gen_A, opt_gen, config.CHECKPOINT_GEN_A)
            save_checkpoint(disc_B, opt_disc, config.CHECKPOINT_DISC_B)
            save_checkpoint(disc_A, opt_disc, config.CHECKPOINT_DISC_A)
if __name__ == "__main__":
    main()