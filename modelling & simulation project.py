import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

output_path = "generated_traffic.csv"
file_path = "CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv"
df = pd.read_csv(file_path).sample(n=100000, random_state=42)

# IP splitting
for direction in ['orig', 'resp']:
    ip_parts = df[f'id.{direction}_h'].str.split('.', expand=True).astype(int)
    ip_parts.columns = [f'id.{direction}h{i+1}' for i in range(4)]
    df.drop(columns=[f'id.{direction}_h'], inplace=True)
    df = pd.concat([df, ip_parts], axis=1)

# Encode categorical columns
cat_cols = ['proto', 'conn_state', 'label', 'history']
label_encoders = {col: LabelEncoder() for col in cat_cols}
for col in cat_cols:
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Clean up
df['id.resp_p'] = pd.to_numeric(df['id.resp_p'], errors='coerce').fillna(0).astype(int)
cols_to_drop = ['detailed-label', 'uid', 'ts_orig', 'ts_resp',
                'local_orig', 'local_resp', 'tunnel_parents']
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df[col] = df[col].fillna(0)

# Scale features
num_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Tensors
X_tensor = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
y_tensor = torch.tensor(df['label'].values, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

# Weight initialization 
def moderate_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.4)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Generator & Discriminator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# GAN Setup
latent_dim = 2
gen = Generator(latent_dim, X_tensor.shape[1]).to(device)
disc = Discriminator(X_tensor.shape[1]).to(device)
gen.apply(moderate_weights_init)
disc.apply(moderate_weights_init)

criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(gen.parameters(), lr=1e-3, betas=(0.5, 0.999))
optimizer_D = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Training
epochs = 100
k_steps_for_G = 3
label_smoothing_real = 0.8
label_smoothing_fake = 0.1

print(f"Training GAN with {X_tensor.shape[1]} features...")

for epoch in range(epochs):
    for real_data, _ in dataloader:
        real_data = real_data.to(device) 
        batch_size = real_data.size(0)

        real_data += 0.05 * torch.randn_like(real_data).to(device)

        real_labels = torch.full((batch_size, 1), label_smoothing_real, device=device)
        fake_labels = torch.full((batch_size, 1), label_smoothing_fake, device=device)

        real_labels += 0.05 * torch.rand_like(real_labels).to(device)
        fake_labels += 0.05 * torch.rand_like(fake_labels).to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_output = disc(real_data)
        d_loss_real = criterion(real_output, real_labels)

        noise = torch.randn(batch_size, latent_dim, device=device)  
        fake_data = gen(noise).detach()
        fake_output = disc(fake_data)
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator (k steps)
        for _ in range(k_steps_for_G):
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_data = gen(noise)
            output = disc(fake_data)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("\nGenerating synthetic data...")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

gen.eval()
num_samples_to_generate = 10000
gen_output_feature_names = df.drop(columns=['label']).columns.tolist()

with torch.no_grad():
    noise = torch.randn(num_samples_to_generate, latent_dim, device=device)
    generated_data_scaled_np = gen(noise).cpu().numpy()

df_generated_scaled = pd.DataFrame(generated_data_scaled_np, columns=gen_output_feature_names)

df_generated_unscaled = pd.DataFrame(index=df_generated_scaled.index)
scaler_num_cols_list = num_cols.tolist()

for col_name in gen_output_feature_names:
    if col_name in scaler_num_cols_list:
        idx = scaler_num_cols_list.index(col_name)
        mean_val = scaler.mean_[idx]
        scale_val = scaler.scale_[idx]
        df_generated_unscaled[col_name] = (df_generated_scaled[col_name] * scale_val) + mean_val
    else:
        df_generated_unscaled[col_name] = df_generated_scaled[col_name]

for col in ['proto', 'conn_state', 'history']:
    le = label_encoders[col]
    labels = np.clip(np.round(df_generated_unscaled[col]).astype(int), 0, len(le.classes_) - 1)
    df_generated_unscaled[col] = le.inverse_transform(labels)

for i in range(1, 5):
    df_generated_unscaled[f'id.origh{i}'] = np.round(df_generated_unscaled[f'id.origh{i}']).astype(int).astype(str)
    df_generated_unscaled[f'id.resph{i}'] = np.round(df_generated_unscaled[f'id.resph{i}']).astype(int).astype(str)

df_generated_unscaled['id.orig_h'] = df_generated_unscaled[[f'id.origh{i}' for i in range(1, 5)]].agg('.'.join, axis=1)
df_generated_unscaled['id.resp_h'] = df_generated_unscaled[[f'id.resph{i}' for i in range(1, 5)]].agg('.'.join, axis=1)
df_generated_unscaled.drop(columns=[f'id.origh{i}' for i in range(1, 5)] + [f'id.resph{i}' for i in range(1, 5)], inplace=True)

for col in ['id.resp_p', 'id.orig_p']:
    df_generated_unscaled[col] = np.clip(np.round(df_generated_unscaled[col]).astype(int), 0, 65535)

df_generated_unscaled.to_csv(output_path, index=False)
print(f"Generated synthetic data saved to {output_path}")
print("\nFirst 5 rows of generated data:")
print(df_generated_unscaled.head())

# strengths of GAN in cybersecurity:
# 1) GANs can generate synthetic samples for underrepresented classes, 
#    improving training for IDS
# 2) GANs can learn to generate "stealthy" attack traffic that mimics normal behavior,
#    which can be used to test and improve IDS robustness
# 3) GANs can produce realistic traffic that reflects statistical properties of real data without exposing sensitive user information
# 
# weaknesses of GAN in cybersecurity:
# 1) If GANs are trained on outdated or limited datasets, they may fail to generalize to newer threats or evolving traffic patterns
# 2) GANs require significant computational resources and time to train properly 
#    which may be a barrier for real-time or resource-constrained environments