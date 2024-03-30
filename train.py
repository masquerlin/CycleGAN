import torch.nn as nn
import torch
from dataset import HorseZebraDataset
import config
import torch.optim as optim
from tqdm import tqdm
from utils import load_checkpoint, save_checkpoint
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from discriminator_model import Discriminator
from generator_model import generator

def train_fn(disc_h, disc_z, gen_z, gen_h, loader, 
             opt_disc:torch.optim.Adam, opt_gen:torch.optim.Adam, 
             l1, mse, d_scaler:torch.cuda.amp.GradScaler, g_scaler:torch.cuda.amp.GradScaler):
    loop = tqdm(loader)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE) 
        # disc_h and disc_z
        with torch.cuda.amp.autocast():
            fake_horse = gen_h(zebra)
            disc_horse_real = disc_h(horse)
            disc_horse_fake = disc_h(fake_horse.detach())
            disc_horse_real_loss = mse(disc_horse_real, torch.ones_like(disc_horse_real))
            disc_horse_fake_loss = mse(disc_horse_fake, torch.zeros_like(disc_horse_fake))
            d_h_loss = disc_horse_real_loss + disc_horse_fake_loss

            fake_zebra = gen_z(horse)
            disc_zebra_real = disc_z(zebra)
            disc_zebra_fake = disc_z(fake_zebra.detach())
            disc_zebra_real_loss = mse(disc_zebra_real, torch.ones_like(disc_zebra_real))
            disc_zebra_fake_loss = mse(disc_zebra_fake, torch.zeros_like(disc_zebra_fake))
            d_z_loss = disc_zebra_real_loss + disc_zebra_fake_loss

            d_loss = (d_h_loss + d_z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        #generator
        #adveisial loss
        d_zebra_fake = disc_z(fake_zebra)
        d_horse_fake = disc_h(fake_horse)
        g_zebra_loss = mse(d_zebra_fake, torch.ones_like(d_zebra_fake))
        g_horse_loss = mse(d_horse_fake, torch.ones_like(d_horse_fake))

        #cycle loss
        cycle_horse = gen_h(fake_zebra)
        cycle_zebra = gen_z(fake_horse)
        cycle_horse_loss = l1(horse, cycle_horse)
        cycle_zebra_loss = l1(zebra, cycle_zebra)

        #identity_loss
        identity_horse = gen_h(horse)
        identity_zebra = gen_z(zebra)
        identity_zebra_loss = l1(horse, identity_horse)
        identity_horse_loss = l1(zebra, identity_zebra)

        g_loss = (
            g_zebra_loss + g_horse_loss +
            cycle_horse_loss * config.LAMBDA_CYCLE +
            cycle_zebra_loss * config.LAMBDA_CYCLE +
            identity_horse_loss * config.LAMBDA_IDENTITY +
            identity_zebra_loss * config.LAMBDA_IDENTITY

        )
        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 ==0:
            save_image(fake_horse*0.5+0.5, f"save_images/horse_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"save_images/zebra_{idx}.png")





def main():
    disc_h = Discriminator(in_channels=3).to(config.DEVICE)
    disc_z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_z = generator(img_channels=3).to(config.DEVICE)
    gen_h = generator(img_channels=3).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_h.parameters()) + list(disc_z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    opt_gen = optim.Adam(
        list(gen_z.parameters()) + list(gen_h.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_h, opt_gen, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_z, opt_gen, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_CRITIC_H, disc_h, opt_disc, config.LEARNING_RATE),
        load_checkpoint(config.CHECKPOINT_CRITIC_Z, disc_z, opt_disc, config.LEARNING_RATE)

    dataset = HorseZebraDataset(config.TRAIN_DIR + 'zebra', config.VAL_DIR + 'horse', transform=config.transforms)

    loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,pin_memory=True)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn()
        if config.SAVE_MODEL:
            save_checkpoint(gen_h, opt_gen, config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_z, opt_gen, config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_h, opt_disc, config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_z, opt_gen, config.CHECKPOINT_CRITIC_Z)
if __name__ == "__main__":
    main()