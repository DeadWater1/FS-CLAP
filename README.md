# FS-CLAP

This repository provides FS-CLAP: fusion-separation clap for emotional speaking style speech retrieval. 


## About this project
This is an opensource project. We adopt the codebase of open_clip and laion_clip for this project.


## Quick start
```python
pip install -r requirements.txt
pip install -e .
```

Then you can follow this script.
```python
import torch
from models import CLAP_Dual
import torch.nn.functional as F
from dataset import audio_process


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLAP_Dual(layers=2, device=device)

ckpt_path = 'save_checkpoints/fsclap_ts/final.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()


text = [
        "A male speaker employs a deep voice to converse naturally with an average speaking tempo and normal vigor."
    ]
audio = [
        'demo1.wav',
        'demo2.wav',
        'demo3.wav',
        'demo4.wav',
        'demo5.wav',
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
```

or directly run the infer script
```python
python src/infer.py
```


## Training

The training settings are given in <u>src/config/train.yaml</u>. Simply run the code by
```
python src/train.py
```


## Datset
TextrolSpeech: https://github.com/jishengpeng/TextrolSpeech
GigaSpeech: https://github.com/SpeechColab/GigaSpeech
Zhvoice: https://github.com/fighting41love/zhvoice
PromptSpeech: https://speechresearch.github.io/prompttts/#promptspeech












