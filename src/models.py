import sys
import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import logging
from collections import OrderedDict

import laion_clap
from loss import AudioTextContrastiveLoss_HN



class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1):
		super().__init__()

		# all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
		self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu1 = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.relu2 = nn.ReLU(inplace=True)

		self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

		self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu3 = nn.ReLU(inplace=True)

		self.downsample = None
		self.stride = stride

		if stride > 1 or inplanes != planes * Bottleneck.expansion:
			# downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
			self.downsample = nn.Sequential(OrderedDict([
				("-1", nn.AvgPool2d(stride)),
				("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
				("1", nn.BatchNorm2d(planes * self.expansion))
			]))

	def forward(self, x: torch.Tensor):
		identity = x

		out = self.relu1(self.bn1(self.conv1(x)))
		out = self.relu2(self.bn2(self.conv2(out)))
		out = self.avgpool(out)
		out = self.bn3(self.conv3(out))

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu3(out)
		return out


class AttentionPool2d(nn.Module):
	def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
		super().__init__()
		self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
		self.k_proj = nn.Linear(embed_dim, embed_dim)
		self.q_proj = nn.Linear(embed_dim, embed_dim)
		self.v_proj = nn.Linear(embed_dim, embed_dim)
		self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
		self.num_heads = num_heads

	def forward(self, x):
		x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
		x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
		x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
		x, _ = F.multi_head_attention_forward(
			query=x[:1], key=x, value=x,
			embed_dim_to_check=x.shape[-1],
			num_heads=self.num_heads,
			q_proj_weight=self.q_proj.weight,
			k_proj_weight=self.k_proj.weight,
			v_proj_weight=self.v_proj.weight,
			in_proj_weight=None,
			in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
			bias_k=None,
			bias_v=None,
			add_zero_attn=False,
			dropout_p=0,
			out_proj_weight=self.c_proj.weight,
			out_proj_bias=self.c_proj.bias,
			use_separate_proj_weight=True,
			training=self.training,
			need_weights=False
		)
		return x.squeeze(0)


class ModifiedResNet(nn.Module):
	"""
	A ResNet class that is similar to torchvision's but contains the following changes:
	- There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
	- Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
	- The final pooling layer is a QKV attention instead of an average pool
	"""

	def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
		super().__init__()
		self.output_dim = output_dim
		self.input_resolution = input_resolution

		# the 3-layer stem
		self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(width // 2)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(width // 2)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(width)
		self.relu3 = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(2)

		# residual layers
		self._inplanes = width  # this is a *mutable* variable used during construction
		self.layer1 = self._make_layer(width, layers[0])
		self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
		self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
		self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

		embed_dim = width * 32  # the ResNet feature dimension
		self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

	def _make_layer(self, planes, blocks, stride=1):
		layers = [Bottleneck(self._inplanes, planes, stride)]

		self._inplanes = planes * Bottleneck.expansion
		for _ in range(1, blocks):
			layers.append(Bottleneck(self._inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		def stem(x):
			x = self.relu1(self.bn1(self.conv1(x)))
			x = self.relu2(self.bn2(self.conv2(x)))
			x = self.relu3(self.bn3(self.conv3(x)))
			x = self.avgpool(x)
			return x

		x = x.type(self.conv1.weight.dtype)
		x = stem(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.attnpool(x)

		return x


class LayerNorm(nn.LayerNorm):
	"""Subclass torch's LayerNorm to handle fp16."""

	def forward(self, x: torch.Tensor):
		orig_type = x.dtype
		ret = super().forward(x.type(torch.float32))
		return ret.type(orig_type)


class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
	def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
		super().__init__()

		self.attn = nn.MultiheadAttention(d_model, n_head)
		self.ln_1 = LayerNorm(d_model)
		self.mlp = nn.Sequential(OrderedDict([
			("c_fc", nn.Linear(d_model, d_model * 4)),
			("gelu", QuickGELU()),
			("c_proj", nn.Linear(d_model * 4, d_model))
		]))
		self.ln_2 = LayerNorm(d_model)
		self.attn_mask = attn_mask

	def attention(self, x: torch.Tensor):
		self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
		return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

	def forward(self, x: torch.Tensor):
		x = x + self.attention(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class Transformer(nn.Module):
	def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
		super().__init__()
		self.width = width
		self.layers = layers
		self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

	def forward(self, x: torch.Tensor):
		return self.resblocks(x)


class AttentionPool(nn.Module):
	def __init__(self, hidden_size=768):
		super().__init__()
		self.attention = nn.Linear(hidden_size, 1)

	def forward(self, x):  # x: [bs,77,768]
		weights = F.softmax(self.attention(x), dim=1)  # [bs,77,1]
		output = torch.sum(weights * x, dim=1)  # [bs,768]
		return output
	

class CLAP_Dual(nn.Module):
    def __init__(self,
                 layers=1,
                 device="cuda" if torch.cuda.is_available() else "cpu"
                ):
        super().__init__()

        self.device = device

        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.loading_checkpoints()
        self.model.to(self.device)

        self.text_branch_type = self.model.model.text_branch_type
        self.text_branch = self.model.model.text_branch
        self.text_projection = self.model.model.text_projection

        self.audio_branch = self.model.model.audio_branch
        self.audio_projection = self.model.model.audio_projection

        self.logit_scale_a = self.model.model.logit_scale_a
        self.logit_scale_t = self.model.model.logit_scale_t

        self.loss = AudioTextContrastiveLoss_HN()

        self.dual = Dual_Adapter(768, layers=layers).to(self.device)


    def loading_checkpoints(self):
        self.model.load_ckpt()

    def encode_text(self, text, device):
        if self.text_branch_type == "transformer":
            text = text.to(device=device, non_blocking=True)
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_branch(x, attn_mask=self.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            y = x

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = self.text_projection(x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        elif self.text_branch_type == "bert":
            # text = self.list_of_dict_of_tensor2dict_of_tensor(text, device)
            # text = BatchEncoding(text)
            x = self.text_branch(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
                token_type_ids=text["token_type_ids"].to(
                    device=device, non_blocking=True
                ),
                output_hidden_states=True
            )["pooler_output"]

            y = x['last_hidden_state']
            # x = self.text_projection(x)
        elif self.text_branch_type == "roberta":
            x = self.text_branch(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
                output_hidden_states=True
            )

            y = x['last_hidden_state']
            x = x["pooler_output"]
            # x = self.text_projection(x)
        elif self.text_branch_type == "bart":
            x = torch.mean(self.text_branch(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
            )["encoder_last_hidden_state"],axis=1)
            y = None
            x = self.text_projection(x)
        else:
            logging.error(f"Model type {self.text_branch_type} not found")
            raise RuntimeError(f"Model type {self.text_branch_type} not found.")
        return x, y

    def get_text_embedding(self, x, tokenizer = None, use_tensor = True):

        if tokenizer is not None:
            text_input = tokenizer(x)
        else:
            text_input = self.model.tokenizer(x)

        device = next(self.model.parameters()).device
        for k in text_input:
            text_input[k] = text_input[k].to(device)
        text_embed, last_hidden_state = self.encode_text(text_input, device=device)
        # text_embed = F.normalize(text_embed, dim=-1)
        # last_hidden_state = F.normalize(last_hidden_state, dim=-1)

        if not use_tensor:
            text_embed = text_embed.detach().cpu().numpy()

        return text_embed, last_hidden_state

    def get_audio_embedding(self, audio_input):
        device = next(self.parameters()).device
        input_dict = {}
        keys = audio_input[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in audio_input], dim=0).to(device)

        outputs = self.audio_branch(input_dict, mixup_lambda=None, device=self.device)
        audio_embed, hidden_state = outputs["embedding"], outputs["fine_grained_embedding"]

        # audio_embed = self.audio_projection(audio_embed)
        # audio_embed = F.normalize(audio_embed, dim=-1)
        # hidden_state = F.normalize(hidden_state, dim=-1)

        return audio_embed, hidden_state

    def forward(self, text, audio_input, embed_reg=True):

        _, text_hidden = self.get_text_embedding(text)
        _, audio_hidden = self.get_audio_embedding(audio_input)

        text_hidden, audio_hidden = self.dual(text_hidden, audio_hidden)
        text_hidden = self.text_projection(text_hidden)
        audio_hidden = self.audio_projection(audio_hidden)

        text_hidden = F.normalize(text_hidden, dim=-1)
        audio_hidden = F.normalize(audio_hidden, dim=-1)

        global_sim_a2t = self.logit_scale_a.exp() * audio_hidden @ text_hidden.T
        global_sim_t2a = self.logit_scale_t.exp() * text_hidden @ audio_hidden.T

        loss = self.loss(global_sim_a2t, global_sim_t2a)

        if embed_reg:
            loss = loss + torch.mean(torch.abs(audio_hidden)) / torch.sqrt(torch.sum(audio_hidden**2)) + \
                torch.mean(torch.abs(text_hidden)) / torch.sqrt(torch.sum(text_hidden**2))
        
        return loss



class Dual_Adapter(nn.Module):
    def __init__(self, width, layers=1, heads=8, attn_mask=None):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        if layers > 0:
            proj_std = (width ** -0.5) * ((2 * layers) ** -0.5)
            attn_std = width ** -0.5
            fc_std = (2 * width) ** -0.5
            for block in self.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.text_ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            QuickGELU(),
            nn.Linear(width * 4, width)
        )

        self.audio_ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            QuickGELU(),
            nn.Linear(width * 4, width)
        )

        self.text_norm = LayerNorm(width)
        self.audio_norm = LayerNorm(width)

        self.text_attn_pool = AttentionPool(width)
        self.audio_attn_pool = AttentionPool(width)


    def inference_audio(self, text_hidden, audio_hidden):

        bs = audio_hidden.shape[0]
        t_seq_len = text_hidden.shape[1]

        text_hidden_norm = self.text_norm(text_hidden).repeat(bs, 1, 1)
        audio_hidden_norm = self.audio_norm(audio_hidden)

        all_hidden = torch.cat([text_hidden_norm, audio_hidden_norm], dim=1)
        all_hidden_output = self.resblocks(all_hidden)
        audio_hidden_output = all_hidden_output[:, t_seq_len:, :] + audio_hidden

        audio_hidden_output = audio_hidden_output + self.audio_ffn(self.audio_norm(audio_hidden_output))
        audio_hidden_output = self.audio_attn_pool(audio_hidden_output)

        return audio_hidden_output
    

    def inference_text(self, text_hidden):

        text_hidden_norm = self.text_norm(text_hidden)
        text_hidden_output = text_hidden_norm + self.text_ffn(text_hidden_norm)

        text_hidden_output = self.text_attn_pool(text_hidden_output)

        return text_hidden_output
    

    def forward(self, text_hidden, audio_hidden):
        
        t_seq_len = text_hidden.shape[1]

        text_hidden_norm = self.text_norm(text_hidden)
        audio_hidden_norm = self.audio_norm(audio_hidden)

        all_hidden = torch.cat([text_hidden_norm, audio_hidden_norm], dim=1)
        all_hidden_output = self.resblocks(all_hidden)

        text_hidden_output = all_hidden_output[:, :t_seq_len, :] + text_hidden
        audio_hidden_output = all_hidden_output[:, t_seq_len:, :] + audio_hidden

        text_hidden_output = text_hidden_output + self.text_ffn(self.text_norm(text_hidden_output))
        audio_hidden_output = audio_hidden_output + self.audio_ffn(self.audio_norm(audio_hidden_output))

        text_hidden_output = self.text_attn_pool(text_hidden_output)
        audio_hidden_output = self.audio_attn_pool(audio_hidden_output)

        return text_hidden_output, audio_hidden_output
    

    def infer(self, text_hidden, audio_hidden):

        bs_t = text_hidden.shape[0]
        bs_a = audio_hidden.shape[0]

        if bs_t != bs_a:
            if bs_t > bs_a:
                repeat_times = bs_t // bs_a
                audio_hidden = audio_hidden.repeat(repeat_times, 1, 1)
            else:
                repeat_times = bs_a // bs_t
                text_hidden = text_hidden.repeat(repeat_times, 1, 1)

        t_seq_len = text_hidden.shape[1]

        text_hidden_norm = self.text_norm(text_hidden)
        audio_hidden_norm = self.audio_norm(audio_hidden)

        all_hidden = torch.cat([text_hidden_norm, audio_hidden_norm], dim=1)
        all_hidden_output = self.resblocks(all_hidden)

        text_hidden_output = all_hidden_output[:, :t_seq_len, :] + text_hidden
        audio_hidden_output = all_hidden_output[:, t_seq_len:, :] + audio_hidden

        text_hidden_output = text_hidden_output + self.text_ffn(self.text_norm(text_hidden_output))
        audio_hidden_output = audio_hidden_output + self.audio_ffn(self.audio_norm(audio_hidden_output))

        text_hidden_output = self.text_attn_pool(text_hidden_output)
        audio_hidden_output = self.audio_attn_pool(audio_hidden_output)

        return text_hidden_output, audio_hidden_output





  