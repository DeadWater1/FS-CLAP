import sys
import os
from pathlib import Path
import random
import numpy as np

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import pandas as pd
import librosa
from tqdm import tqdm
from clap.utils.data import get_audio_features, int16_to_float32, float32_to_int16
from clap.utils.text_transform import text_preprocess



tqdm.pandas(desc="文件是否存在") 


def display_trainable_params(model):
    """Display the trainable parameters of the model."""
    import pandas as pd
    from IPython.display import display

    trainable_params = {
        "name": [],
        "shape": [],
        "dtype": [],
        "params": [],
    }

    frozenable_params = {
        "name": [],
        "shape": [],
        "dtype": [],
        "params": [],
    }

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params["name"].append(name)
            trainable_params["shape"].append(param.shape)
            trainable_params["dtype"].append(param.dtype)
            trainable_params["params"].append(param.numel())

        if not param.requires_grad:
            frozenable_params["name"].append(name)
            frozenable_params["shape"].append(param.shape)
            frozenable_params["dtype"].append(param.dtype)
            frozenable_params["params"].append(param.numel())

    with pd.option_context("display.max_colwidth", None):
        print("Trainable parameters in the model:")
        display(pd.DataFrame(trainable_params))
        
        print("Total trainable params:", round(sum(trainable_params["params"]) / 1e6, 2), 'M')
        print("Total frozenable params:", round(sum(frozenable_params["params"]) / 1e6, 2), 'M')
    
    return trainable_params, frozenable_params


class TrainDataset(Dataset):
    def __init__(self, csv_paths):
        
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        
        dfs = []
        for path in csv_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV files does not exist: {path}")
            df = pd.read_csv(path)

            required_columns = ['wav_fp', 'style_prompt']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"CSV files {path} have not: {col}")
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        full_paths = df['wav_fp']
        mask = full_paths.progress_apply(os.path.exists)
        self.df = df[mask].reset_index(drop=True)

        print(f"Original dataset size: {len(df)}")
        print(f"Filtered dataset size: {len(self.df)}")

        self.wav_path = self.df['wav_fp']
        self.style_prompt = self.df['style_prompt']

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        wav_fp = self.wav_path[idx]

        return {
            'wav_fp': wav_fp,
            'style_prompt': self.style_prompt[idx],
        }


model_cfg = {
    'audio_length': 1024, 
    'clip_samples': 480000, 
    'mel_bins': 64, 
    'sample_rate': 48000, 
    'window_size': 1024, 
    'hop_size': 480, 
    'fmin': 50, 
    'fmax': 14000, 
    'class_num': 527, 
    'model_type': 'HTSAT', 
    'model_name': 'tiny'
}


def collate_fn(batch):
    wav_fp = [item['wav_fp'] for item in batch]
    style_prompt = [item['style_prompt'] for item in batch]

    audio_input = []
    for f in wav_fp:
        audio_waveform, _ = librosa.load(f, sr=48000)           
        audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        temp_dict = {}
        temp_dict = get_audio_features(
            temp_dict, audio_waveform, 480000, 
            data_truncating='rand_trunc', 
            data_filling='repeatpad',
            audio_cfg=model_cfg,
            require_grad=audio_waveform.requires_grad
        )
        audio_input.append(temp_dict)

    return {
        'style_prompt': style_prompt,
        'audio_input': audio_input
    }


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def evaluate(model, dataloader, writer, global_step, split_type):

    model.eval()

    all_audio_features = []
    all_text_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_data = batch['style_prompt']
            audio_input = batch['audio_input']

            text_features, _ = model.get_text_embedding(text_data, use_tensor=True)
            audio_features, _ = model.get_audio_embedding(audio_input)

            all_audio_features.append(audio_features.detach().cpu())
            all_text_features.append(text_features.detach().cpu())

        all_audio_features = torch.cat(all_audio_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

        logit_scale_a = model.logit_scale_a.exp().cpu()
        logit_scale_t = model.logit_scale_t.exp().cpu()


    logits_per_audio = logit_scale_a * all_audio_features @ all_text_features.t()
    logits_per_text = logit_scale_t * all_text_features @ all_audio_features.t()


    audio_diag = torch.diag(logits_per_audio)
    text_diag = torch.diag(logits_per_text)

    ranks_a2t = (logits_per_audio > audio_diag.view(-1, 1)).sum(dim=1)
    ranks_t2a = (logits_per_text > text_diag.view(-1, 1)).sum(dim=1)
    
    metrics = {}
    
    for k in [1, 5, 10]:
        metrics[f"{split_type}/audio_to_text_R@{k}"] = (ranks_a2t < k).float().mean().item()
        metrics[f"{split_type}/text_to_audio_R@{k}"] = (ranks_t2a < k).float().mean().item()


    ranks_a2t_np = ranks_a2t.numpy()
    ranks_t2a_np = ranks_t2a.numpy()
    
    metrics[f"{split_type}/audio_to_text_mAP@10"] = np.mean(
        np.where(ranks_a2t_np < 10, 1 / (ranks_a2t_np + 1), 0.0)
    )
    metrics[f"{split_type}/text_to_audio_mAP@10"] = np.mean(
        np.where(ranks_t2a_np < 10, 1 / (ranks_t2a_np + 1), 0.0)
    )

    for name, val in metrics.items():
        writer.add_scalar(name, val, global_step)
        
    writer.flush()
    model.train()



def evaluate_modify(model, dataloader, writer, global_step, split_type='test'):

    model.eval()

    all_audio_features = []
    all_text_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_data = batch['style_prompt']
            audio_input = batch['audio_input']

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

        logit_scale_a = model.logit_scale_a.exp().cpu()
        logit_scale_t = model.logit_scale_t.exp().cpu()


    logits_per_audio = logit_scale_a * all_audio_features @ all_text_features.t()
    logits_per_text = logit_scale_t * all_text_features @ all_audio_features.t()

    diag_a2t = torch.diag(logits_per_audio)
    diag_t2a = torch.diag(logits_per_text)

    ranks_a2t = (logits_per_audio > diag_a2t.view(-1, 1)).sum(dim=1)
    ranks_t2a = (logits_per_text > diag_t2a.view(-1, 1)).sum(dim=1)
    
    metrics = {}
    
    for k in [1, 5, 10]:
        metrics[f"{split_type}/audio_to_text_R@{k}"] = (ranks_a2t < k).float().mean().item()
        metrics[f"{split_type}/text_to_audio_R@{k}"] = (ranks_t2a < k).float().mean().item()


    ranks_a2t_np = ranks_a2t.numpy()
    ranks_t2a_np = ranks_t2a.numpy()
    
    metrics[f"{split_type}/audio_to_text_mAP@10"] = np.mean(
        np.where(ranks_a2t_np < 10, 1 / (ranks_a2t_np + 1), 0.0)
    )
    metrics[f"{split_type}/text_to_audio_mAP@10"] = np.mean(
        np.where(ranks_t2a_np < 10, 1 / (ranks_t2a_np + 1), 0.0)
    )


    for name, val in metrics.items():
        writer.add_scalar(name, val, global_step)
        
    writer.flush()
    model.train()


class MGACLAP_Dataset(Dataset):

    def __init__(self, csv_paths):

        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        
        dfs = []
        for path in csv_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV files does not exist: {path}")
            df = pd.read_csv(path)

            required_columns = ['wav_fp', 'style_prompt']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"CSV files {path} have not: {col}")
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        full_paths = df['wav_fp']
        mask = full_paths.progress_apply(os.path.exists)
        self.df = df[mask].reset_index(drop=True)

        print(f"Original dataset size: {len(df)}")
        print(f"Filtered dataset size: {len(self.df)}")

        self.wav_path = self.df['wav_fp']
        self.style_prompt = self.df['style_prompt']

        self.max_length = 320000

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        wav_path = self.wav_path[index]

        waveform, _ = librosa.load(wav_path, sr=32000, mono=True, duration=300)
        if self.max_length != 0:
            # if audio length is longer than max_length, we randomly crop it to mac length
            if waveform.shape[-1] > self.max_length:
                max_start = waveform.shape[-1] - self.max_length
                start = random.randint(0, max_start)
                waveform = waveform[start: start + self.max_length]

        caption = text_preprocess(self.style_prompt[index])

        return torch.tensor(waveform), caption



def evaluate_microsoft(model, dataloader, writer, global_step, split_type, device):

    model.eval()

    all_audio_features = []
    all_text_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text = batch['text_tensor']
            audio_input = batch['audio_tensor']

            preprocessed_audio = audio_input.squeeze(1).to(device)
            preprocessed_text = {key: value.squeeze(1).to(device) for key, value in text.items()}

            text_features = model._get_text_embeddings(preprocessed_text)
            audio_features = model._get_audio_embeddings(preprocessed_audio)

            all_audio_features.append(audio_features.detach().cpu())
            all_text_features.append(text_features.detach().cpu())

        all_audio_features = torch.cat(all_audio_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

        all_text_features = F.normalize(all_text_features, dim=-1)
        all_audio_features = F.normalize(all_audio_features, dim=-1)

        logit_scale = model.logit_scale.exp().cpu()


    logits_per_audio = logit_scale * all_audio_features @ all_text_features.t()
    logits_per_text = logit_scale * all_text_features @ all_audio_features.t()

    audio_diag = torch.diag(logits_per_audio)
    text_diag = torch.diag(logits_per_text)

    ranks_a2t = (logits_per_audio > audio_diag.view(-1, 1)).sum(dim=1)
    ranks_t2a = (logits_per_text > text_diag.view(-1, 1)).sum(dim=1)
    
    metrics = {}
    
    for k in [1, 5, 10]:
        metrics[f"{split_type}/audio_to_text_R@{k}"] = (ranks_a2t < k).float().mean().item()
        metrics[f"{split_type}/text_to_audio_R@{k}"] = (ranks_t2a < k).float().mean().item()


    ranks_a2t_np = ranks_a2t.numpy()
    ranks_t2a_np = ranks_t2a.numpy()
    
    metrics[f"{split_type}/audio_to_text_mAP@10"] = np.mean(
        np.where(ranks_a2t_np < 10, 1 / (ranks_a2t_np + 1), 0.0)
    )
    metrics[f"{split_type}/text_to_audio_mAP@10"] = np.mean(
        np.where(ranks_t2a_np < 10, 1 / (ranks_t2a_np + 1), 0.0)
    )

    for name, val in metrics.items():
        writer.add_scalar(name, val, global_step)
        
    writer.flush()
    model.train()


def test_collate_fn2(batch):

    wav_fp = [item['wav_fp'] for item in batch]
    style_prompt = [item['style_prompt'] for item in batch]

    audio_input = []
    for f in wav_fp:
        audio_waveform, _ = librosa.load(f, sr=48000)           
        audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        temp_dict = {}
        temp_dict = get_audio_features(
            temp_dict, audio_waveform, 480000, 
            data_truncating='rand_trunc', 
            data_filling='repeatpad',
            audio_cfg=model_cfg,
            require_grad=audio_waveform.requires_grad
        )
        audio_input.append(temp_dict)

    return {
        'style_prompt': style_prompt,
        'audio_input': audio_input,
    }


def audio_process(wav_fp):

    audio_input = []
    for f in wav_fp:
        audio_waveform, _ = librosa.load(f, sr=48000)           
        audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        temp_dict = {}
        temp_dict = get_audio_features(
            temp_dict, audio_waveform, 480000, 
            data_truncating='rand_trunc', 
            data_filling='repeatpad',
            audio_cfg=model_cfg,
            require_grad=audio_waveform.requires_grad
        )
        audio_input.append(temp_dict)

    return audio_input





