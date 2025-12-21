import torch
import torch.nn.functional as F

from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig


class VisionEmbedding: 
    def __init__(self, repo, device="cpu"):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.vision_model = ColQwen2.from_pretrained(
            repo,
            quantization_config=bnb_config, # Apply 4-bit config
            torch_dtype=torch.bfloat16,
            device_map=device, # Automatically map to GPU
            trust_remote_code=True
        )
        self.vision_model.eval()

        self.vision_processor = ColQwen2Processor.from_pretrained(repo, trust_remote_code=True)

    def encode(self, input, input_type):
        if input_type == "text":
            processed_input = self.vision_processor.process_queries([input])
        else:
            processed_input = self.vision_processor.process_images([input])
        
        processed_input = {k: v.to(self.vision_model.device) for k, v in processed_input.items()}

        # Generate embedding
        with torch.no_grad():
            embedding = self.vision_model(**processed_input)
            embedding = embedding.cpu().float().numpy().tolist()

        return embedding


class TextEmbedding:
    def __init__(self, repo, device="cpu"):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(repo)


        self.model = AutoModel.from_pretrained(
            repo,
            quantization_config=bnb_config,
            device_map=device # Let accelerate handle device placement
        )
        self.model.eval() # Set model to evaluation mode

    def encode(self, input):

        inputs = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().tolist()
        
        return embeddings