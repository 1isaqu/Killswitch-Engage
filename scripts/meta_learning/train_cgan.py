import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from cgan_model import Generator, Discriminator

# Configurações de Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

class CGANDataset(Dataset):
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        # Assuming the label is 'best_threshold'
        self.labels = df['best_threshold'].values.astype(np.float32)
        # Drop identifiers and labels to get just the condition features
        cols_to_drop = ['usuario_id', 'best_threshold', 'best_precision', 'index']
        features = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        self.conditions = features.values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.conditions[idx]), torch.tensor(self.labels[idx]).unsqueeze(-1)


def load_config():
    with open(BASE_DIR / "config_cGAN.yaml", "r") as f:
        return yaml.safe_load(f)

def plot_losses(losses_d, losses_g, epochs):
    os.makedirs(LOG_DIR / "losses", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), losses_d, label="Discriminator Loss")
    plt.plot(range(epochs), losses_g, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(LOG_DIR / "losses" / "training_curves.png")
    plt.close()

def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train():
    config = load_config()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR / "gradients", exist_ok=True)

    # Hiperparâmetros
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    latent_dim = config["latent_dim"]
    n_critic = config["n_critic"]
    c_dim = config["condition_dim"]
    
    # Dataset
    train_dataset = CGANDataset(DATA_DIR / "cgan_users_train.parquet")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # A dimensão atual da condição vem do Dataset
    real_c_dim = train_dataset.conditions.shape[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iniciar Redes
    generator = Generator(
        latent_dim=latent_dim, 
        condition_dim=real_c_dim, 
        hidden_dim=config["hidden_dim"], 
        dropout_rate=config["dropout_rate"], 
        num_layers=config["hidden_layers"]
    ).to(device)

    discriminator = Discriminator(
        condition_dim=real_c_dim, 
        hidden_dim=config["hidden_dim"], 
        dropout_rate=config["dropout_rate"], 
        num_layers=config["hidden_layers"]
    ).to(device)
    
    # Losses
    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_reg = nn.L1Loss() # Regressão encoraja o GAN a não fugir muito da realidade (L1/MAE)
    
    # Otimizadores (Adam com fator TTUR)
    opt_G = optim.Adam(generator.parameters(), lr=config["learning_rate_g"], betas=tuple(config["betas"]))
    opt_D = optim.Adam(discriminator.parameters(), lr=config["learning_rate_d"], betas=tuple(config["betas"]))

    # Tracking
    hist_loss_d, hist_loss_g = [], []
    
    print(f"Iniciando treino da CGAN Meta-Observer por {epochs} épocas...")
    print(f"Condição Dinâmica: Vetores de D={real_c_dim}")

    for epoch in range(epochs):
        d_loss_accum = 0.0
        g_loss_accum = 0.0
        
        for i, (condition, real_threshold) in enumerate(train_loader):
            condition = condition.to(device)
            real_threshold = real_threshold.to(device)
            
            cur_batch_size = condition.size(0)
            
            # Ground truth do Discriminador (BCE)
            real_labels = torch.ones(cur_batch_size, 1).to(device)
            fake_labels = torch.zeros(cur_batch_size, 1).to(device)

            # ========================
            #  Treino Discriminador
            # ========================
            for _ in range(n_critic):
                opt_D.zero_grad()
                
                # Ruído simulado
                z = torch.randn(cur_batch_size, latent_dim).to(device)
                fake_threshold = generator(z, condition)
                
                # Previsões
                pred_real = discriminator(real_threshold, condition)
                pred_fake = discriminator(fake_threshold.detach(), condition)
                
                # Losses
                loss_d_real = criterion_adv(pred_real, real_labels)
                loss_d_fake = criterion_adv(pred_fake, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) / 2
                
                loss_d.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), config["gradient_clipping"])
                opt_D.step()

            d_loss_accum += loss_d.item()

            # ========================
            #  Treino Gerador
            # ========================
            opt_G.zero_grad()
            
            z = torch.randn(cur_batch_size, latent_dim).to(device)
            gen_threshold = generator(z, condition)
            
            pred_fake_for_g = discriminator(gen_threshold, condition)
            
            # Loss do gerador: enganar o D + Regressão no threshold real
            adv_loss_val = criterion_adv(pred_fake_for_g, real_labels)
            reg_loss_val = criterion_reg(gen_threshold, real_threshold)
            
            loss_g = (adv_loss_val * config["loss_weights"]["adversarial"]) + \
                     (reg_loss_val * config["loss_weights"]["regression"])
                     
            loss_g.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), config["gradient_clipping"])
            opt_G.step()
            
            g_loss_accum += loss_g.item()

        # Final da Época - Media
        d_loss_avg = d_loss_accum / len(train_loader)
        g_loss_avg = g_loss_accum / len(train_loader)
        hist_loss_d.append(d_loss_avg)
        hist_loss_g.append(g_loss_avg)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            grad_g = compute_gradient_norm(generator)
            grad_d = compute_gradient_norm(discriminator)
            print(f"Epoch {epoch+1:03d}/{epochs} | D Loss: {d_loss_avg:.4f} | G Loss: {g_loss_avg:.4f} | |G_grad|: {grad_g:.3f} | |D_grad|: {grad_d:.3f}")

    # Salvar modelos finais
    torch.save(generator.state_dict(), MODEL_DIR / "generator_final.pth")
    torch.save(discriminator.state_dict(), MODEL_DIR / "discriminator_final.pth")
    
    # Plotar os resultados do treinamento de forma automatizada
    plot_losses(hist_loss_d, hist_loss_g, epochs)
    
    print("Treino da cGAN Meta-Observer Concluído com Sucesso!")
    print(f"Modelos salvos em {MODEL_DIR}")

if __name__ == "__main__":
    train()
