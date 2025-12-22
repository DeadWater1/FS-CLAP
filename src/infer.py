import torch
from models import CLAP_Dual
import torch.nn.functional as F
from dataset import audio_process



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLAP_Dual(layers=2, device=device)

ckpt_path = '/home/beno123/Cjunbiao/QwenLM/save_checkpoints/AudioCLIP/Ours_ts/model_step_19600.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()


text = [
        "A male speaker employs a deep voice to converse naturally with an average speaking tempo and normal vigor."
        # "A male speaker employs a deep voice to converse naturally with an average speaking tempo and normal vigor.",
        # "The joyful woman's high-pitched voice carried a slow yet energetic speech.",
        # "A woman with a high-pitched voice speaks with rapid energy.",
        # "The girl's voice resonates with energy in a high tone.",
        # "His low-pitched voice conveys high energy as the male speaker speaks naturally."
    ]
audio = [
        'data/libritts/train-clean-360/3009/10328/3009_10328_000023_000002.wav',
        'data/libritts/train-clean-100/3235/28452/3235_28452_000007_000001.wav',
        'data/libritts/train-clean-100/3830/12535/3830_12535_000033_000002.wav',
        'data/libritts/train-clean-360/7247/77778/7247_77778_000014_000012.wav',
        'data/VCTK-Corpus/wav48/p307/p307_129.wav',
        'data/VCTK-Corpus/wav48/p304/p304_145.wav',
        'data/emotion_dataset/Emotional Speech Dataset (ESD)/Emotion Speech Dataset/0015/Sad/0015_001348.wav',
        'data/libritts/train-clean-360/459/127522/459_127522_000002_000000.wav',
        'data/emotion_dataset/Emotional Speech Dataset (ESD)/Emotion Speech Dataset/0017/Sad/0017_001060.wav',
        'data/libritts/train-clean-360/1382/130516/1382_130516_000023_000000.wav',
        'data/libritts/train-clean-360/5293/82020/5293_82020_000011_000001.wav',
        'data/libritts/train-clean-360/1413/121799/1413_121799_000034_000005.wav',
        'data/emotion_dataset/MEAD/W028/audio/happy/level_2/020.wav',
    ]





with torch.no_grad():
    _, text_features_raw = model.get_text_embedding(text, use_tensor=True)
    _, audio_features_raw = model.get_audio_embedding(audio_process(audio))

    text_features, audio_features = model.dual.infer(text_features_raw, audio_features_raw)

    text_features = model.text_projection(text_features)
    audio_features = model.audio_projection(audio_features)

    output_audio_features = F.normalize(audio_features, dim=-1)
    output_text_features = F.normalize(text_features, dim=-1)[0]

    out = output_text_features @ output_audio_features.T
    # Scale the logits to make the probability distribution sharper
    probs = (out * 50).softmax(dim=-1).cpu().numpy()

print(probs)











