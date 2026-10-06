import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer

class CocoFormatDataset(Dataset):
    """
    Simplified JSON Dataset loader.
    Reads a JSON list containing dictionaries with 'imagePath' and 'gt' keys.
    """

    def __init__(self, args, ann_file, img_prefix, preprocess=None):
        self.args = args
        self.img_prefix = img_prefix
        self.preprocess = preprocess

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Prepare static steering prompt embeddings
        prompt_enc = self.tokenizer(args.streeing_prompt, return_tensors="pt")
        self.prompt_ids = prompt_enc.input_ids.squeeze(0)
        self.prompt_mask = prompt_enc.attention_mask.squeeze(0)

        # Load dataset JSON
        with open(ann_file, 'r') as f:
            self.raw_data = json.load(f)

        self.db = self._load_dataset()
        print(f"=> Loaded {len(self.db)} samples from {ann_file}")

    def _load_dataset(self):
        db = []
        for item in self.raw_data:
            image_rel_path = item["imagePath"]
            image_file = os.path.join(self.img_prefix, image_rel_path)
            caption = item["gt"]

            # The caption string ends without an EOS token in the raw JSON, so we add it if needed
            if not caption.endswith(self.tokenizer.eos_token):
                caption = caption + " " + self.tokenizer.eos_token

            # Extract category_name from the caption (it is inside quotes)
            # Example: It is a species of "Black footed Albatross" as it has...
            import re
            match = re.search(r'It is a species of "([^"]+)"', caption)
            if match:
                category_name_raw = match.group(1)
                category_name = f'"{category_name_raw}"'
            else:
                category_name_raw = "Unknown"
                category_name = '"Unknown"'

            prefix_str = "It is a species of "

            # Character span of the (quoted) category name inside the caption.
            span_start = len(prefix_str)
            span_end = span_start + len(category_name)

            target_enc = self.tokenizer(
                caption, 
                padding="max_length", 
                truncation=True,
                max_length=self.args.max_seq_len, 
                return_tensors="pt",
                return_offsets_mapping=True
            )

            target_ids = target_enc.input_ids.squeeze(0)
            target_mask = target_enc.attention_mask.squeeze(0)

            # Tokenizer-agnostic category mask: a token belongs to the category
            # if its character offsets overlap the category span.
            offsets = target_enc["offset_mapping"].squeeze(0)
            category_mask = (
                (offsets[:, 0] < span_end) & (offsets[:, 1] > span_start)
            ).long()

            db.append({
                "image_file": image_file,
                "prompt_ids": self.prompt_ids,
                "prompt_mask": self.prompt_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "category_name": category_name_raw,
                "caption": caption,
                "category_mask": category_mask
            })

        return db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        sample = self.db[idx]

        with Image.open(sample["image_file"]) as cv_image:
            cv_image = cv_image.convert("RGB")
            if self.preprocess is not None:
                image = self.preprocess(cv_image)
            else:
                image = cv_image

        return {
            "image": image,
            "prompt_ids": sample["prompt_ids"],
            "prompt_mask": sample["prompt_mask"],
            "target_ids": sample["target_ids"],
            "target_mask": sample["target_mask"],
            "caption": sample["caption"],
            "category_name": sample["category_name"],
            "category_mask": sample["category_mask"]
        }
