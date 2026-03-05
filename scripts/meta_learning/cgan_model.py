import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Gerador da cGAN (Meta-Usuário).
    Recebe um ruído aleatório e o embedding do usuário (condição) 
    e gera uma sugestão de threshold contínuo [0.2, 0.8].
    """
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int, dropout_rate: float, num_layers: int = 3):
        super().__init__()
        
        # A entrada do Gerador é a concatenação do ruído latente (Z) com o Perfil do Usuário (C)
        input_dim = latent_dim + condition_dim
        
        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_dim))
            in_features = hidden_dim
            
        self.net = nn.Sequential(*layers)
        
        # Última camada gera o Threshold. Usamos Tanh e re-escalamos.
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Tanh comprime o output para [-1, 1]
        )

    def forward(self, noise, condition):
        # noise: (batch_size, latent_dim)
        # condition: (batch_size, condition_dim)
        x = torch.cat([noise, condition], dim=-1)
        features = self.net(x)
        out = self.out_layer(features)
        
        # Remapeando de [-1, 1] (Tanh) para a faixa de thresholds úteis [0.3, 0.8] approx
        # Tanh varia de -1 a 1. 
        # range = 0.8 - 0.2 = 0.6. center = 0.5.
        # out * 0.3 (range [-0.3, 0.3]) + 0.5 = range [0.2, 0.8]
        threshold = out * 0.30 + 0.55 
        return threshold


class Discriminator(nn.Module):
    """
    Discriminador da cGAN.
    Recebe um threshold proposto (1D) e a condição do usuário 
    e tenta discernir se esse threshold é o ótimo (Real) ou o gerado (Fake).
    """
    def __init__(self, condition_dim: int, hidden_dim: int, dropout_rate: float, num_layers: int = 3):
        super().__init__()
        
        # A entrada é o threshold analisado + vetor de perfis
        input_dim = 1 + condition_dim
        
        layers = []
        in_features = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim // (2**i))) # Reduzindo neurônios em funil
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim // (2**i)
            
        self.net = nn.Sequential(*layers)
        
        # BCEWithLogitsLoss requer logit de saída (sem sigmoid)
        self.out_layer = nn.Linear(in_features, 1)

    def forward(self, threshold, condition):
        # threshold: (batch_size, 1)
        # condition: (batch_size, condition_dim)
        x = torch.cat([threshold, condition], dim=-1)
        features = self.net(x)
        return self.out_layer(features)


class ConditionEncoder(nn.Module):
    """
    Rede auxiliar opcional para comprimir/formatar features brutas do modelo K-Means
    e estatísticas do usuário em um vetor denso fixo (condition_dim).
    """
    def __init__(self, raw_feature_dim: int, condition_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(raw_feature_dim, condition_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(condition_dim * 2, condition_dim),
            nn.LayerNorm(condition_dim) # Estabiliza a condição
        )
        
    def forward(self, raw_features):
        return self.net(raw_features)
