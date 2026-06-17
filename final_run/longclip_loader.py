import torch
import open_clip
from huggingface_hub import hf_hub_download

def load_longclip(model_name='ViT-L-14', pretrained='openai', context_length=248):
    """
    Loads the official LongCLIP weights from Hugging Face if available.
    Falls back to standard open_clip with manually expanded positional embeddings if it fails.
    """
    try:
        print(f"Attempting to load official LongCLIP weights for {model_name}...")
        # 1. Initialize bare model
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
        
        # 2. Expand context length
        model.context_length = context_length
        pe = model.positional_embedding
        new_pe = torch.zeros(context_length, pe.shape[1], device=pe.device, dtype=pe.dtype)
        new_pe[:pe.shape[0]] = pe
        new_pe[pe.shape[0]:] = pe[-1]
        model.positional_embedding = torch.nn.Parameter(new_pe)
        
        if hasattr(model, 'attn_mask') and model.attn_mask is not None:
            mask = torch.empty(context_length, context_length, device=model.attn_mask.device)
            mask.fill_(float("-inf"))
            mask.triu_(1)
            model.attn_mask = mask

        # 3. Download and load official LongCLIP weights
        path = hf_hub_download(repo_id="BeichenZhang/LongCLIP-L", filename="longclip-L.pt")
        sd = torch.load(path, map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"LongCLIP weights loaded successfully. Missing keys: {len(missing)}")
        source = 'longclip_official'
    except Exception as e:
        print(f"Failed to load official LongCLIP weights: {e}")
        print("Falling back to OpenAI weights with manually padded context length...")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        
        model.context_length = context_length
        pe = model.positional_embedding
        new_pe = torch.zeros(context_length, pe.shape[1], device=pe.device, dtype=pe.dtype)
        new_pe[:pe.shape[0]] = pe
        new_pe[pe.shape[0]:] = pe[-1]
        model.positional_embedding = torch.nn.Parameter(new_pe)
        
        if hasattr(model, 'attn_mask') and model.attn_mask is not None:
            mask = torch.empty(context_length, context_length, device=model.attn_mask.device)
            mask.fill_(float("-inf"))
            mask.triu_(1)
            model.attn_mask = mask
        source = 'openai_padded'
        
    return model, preprocess, source

def tokenize(texts, model_name='ViT-L-14', context_length=248):
    try:
        return open_clip.tokenize(texts, context_length=context_length)
    except TypeError:
        tokenizer = open_clip.get_tokenizer(model_name)
        return tokenizer(texts)
