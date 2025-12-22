import os
import umap
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

from src.models import CLAP_Dual
from src.dataset import set_seed
from src.dataset import test_collate_fn2
from torch.utils.data import DataLoader, Dataset




tqdm.pandas()

class TestDataset(Dataset):
    def __init__(self, csv_paths, max_samples=None):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        
        dfs = []
        for path in csv_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file does not exist: {path}")
            df = pd.read_csv(path)

            required_columns = ['wav_fp', 'style_prompt']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"CSV file {path} does not have required column: {col}")
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        full_paths = df['wav_fp']
        # Use progress_apply from tqdm for a progress bar
        mask = full_paths.progress_apply(os.path.exists)
        self.df = df[mask].reset_index(drop=True)

        if max_samples is not None and len(self.df) > max_samples:
            self.df = self.df.sample(n=max_samples, random_state=3407).reset_index(drop=True)
            print(f"Sampled {max_samples} data points for the test set.")

        print(f"Original dataset size: {len(df)}")
        print(f"Filtered dataset size (existing files): {len(self.df)}")

        self.wav_path = self.df['wav_fp']
        self.style_prompt = self.df['style_prompt']      

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'wav_fp': self.wav_path[idx],
            'style_prompt': self.style_prompt[idx],
        }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(3407)

loading_ckpt = 'save_checkpoints/AudioCLIP/Ours_ps/model_step_1520.pt'
model = CLAP_Dual(layers=2, device=device)

if loading_ckpt:
    print(f"Loading checkpoint: {loading_ckpt}")
    checkpoint = torch.load(loading_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
model.eval()




test_csv_path = [
    '/data2/video_all_data/emotion_dataset_en/TextrolSpeech_test.csv'
]
test_dataset = TestDataset(csv_paths=test_csv_path, max_samples=1000)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=64, 
    collate_fn=test_collate_fn2,
    shuffle=False,
    num_workers=4
)

all_audio_features = []
all_text_features = []


for batch in tqdm(test_dataloader, desc="Extracting embeddings"):
    text_data = batch['style_prompt']
    audio_input = batch['audio_input']

    with torch.no_grad():
        _, text_features_raw = model.get_text_embedding(text_data, use_tensor=True)
        _, audio_features_raw = model.get_audio_embedding(audio_input)

        text_features, audio_features = model.dual(text_features_raw, audio_features_raw)
        text_features = model.text_projection(text_features)
        audio_features = model.audio_projection(audio_features)

    all_audio_features.append(audio_features.detach().cpu())
    all_text_features.append(text_features.detach().cpu())

all_audio_features = torch.cat(all_audio_features, dim=0)
all_text_features = torch.cat(all_text_features, dim=0)


all_audio_features = F.normalize(all_audio_features, dim=-1)
all_text_features = F.normalize(all_text_features, dim=-1)

print("Clustering text embeddings using K-Means...")
n_clusters = 3      # setting clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')


cluster_labels = kmeans.fit_predict(all_text_features.numpy())
print(f"Clustering complete. Found {n_clusters} clusters.")



df_audio = pd.DataFrame({
    'cluster_label': cluster_labels,
    'modality': 'Audio'
})
df_text = pd.DataFrame({
    'cluster_label': cluster_labels,
    'modality': 'Text'
})
plot_df = pd.concat([df_audio, df_text], ignore_index=True)



all_embeddings_normalized = np.concatenate([all_audio_features, all_text_features], axis=0)
print("Running UMAP for dimensionality reduction...")
reducer = umap.UMAP(
    n_neighbors=15,
    spread=0.8,
    min_dist=0.8,
    n_components=2,
    metric='cosine',
    random_state=3407
)
embedding_2d = reducer.fit_transform(all_embeddings_normalized)
plot_df['x'] = embedding_2d[:, 0]
plot_df['y'] = embedding_2d[:, 1]
print("UMAP reduction complete.")


print("Generating the visualization...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 10))


sns.scatterplot(
    data=plot_df, 
    x='x', 
    y='y', 
    hue='cluster_label', 
    style='modality',
    palette=sns.color_palette('viridis', n_colors=n_clusters), 
    s=60,
    alpha=0.8,
    ax=ax
)

ax.set_xlabel('') 
ax.set_ylabel('')  
ax.set_xticklabels([])
ax.set_yticklabels([])


ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

ax.grid(False)
ax.spines['bottom'].set_color('black')  
ax.spines['top'].set_color('black')  
ax.spines['right'].set_color('black')  
ax.spines['left'].set_color('black')  

ax.spines['top'].set_linewidth(1.5)  
ax.spines['right'].set_linewidth(1.5)  
ax.spines['bottom'].set_linewidth(1.5)  
ax.spines['left'].set_linewidth(1.5)   

output_filename = f'dual_d{n_clusters}.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nVisualization complete! Image saved as: {output_filename}")