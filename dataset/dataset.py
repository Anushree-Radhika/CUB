import os
import random

from torch.utils.data import Dataset
from pycocotools.coco import COCO

from transformers import GPT2Tokenizer
from PIL import Image


class CocoFormatDataset(Dataset):
    """
    Generic COCO-format dataset loader.

    Images are loaded and preprocessed lazily in __getitem__().
    This avoids loading the entire image dataset into memory during
    dataset initialization.

    Returns:
        - image
        - prompt_ids
        - prompt_mask
        - target_ids
        - target_mask
        - caption
    """

    def __init__(self, args, ann_file, img_prefix, preprocess=None,):

        self.args = args
        self.img_prefix = img_prefix
        self.preprocess = preprocess
        self.drop_parts =args.drop_parts

        # Update the tokenizer in case of a new decoder model.
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.decoder_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt_enc = self.tokenizer(args.streeing_prompt,return_tensors="pt")
        self.prompt_ids = prompt_enc.input_ids.squeeze(0)
        self.prompt_mask = prompt_enc.attention_mask.squeeze(0)

        self.coco = COCO(ann_file)

        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)

        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

        self.categories = {cat["id"]: cat
            for cat in self.coco.loadCats(
                self.coco.getCatIds())}

        self.db = self._load_dataset()

        print(f"=> num_images: {self.num_images}")
        print(f"=> load {len(self.db)} samples")


    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image information.

        Returns:
            id2name: image_id -> filename
            name2id: filename -> image_id
        """

        id2name = {}
        name2id = {}

        for image_id, image in imgs.items():

            file_name = image["filename"]

            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id


    def _load_dataset(self):
        """
        Load dataset metadata only.
        """

        db = []

        for img_id in self.img_ids:
            db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return db


    def get_attributes_for_a_keypoint(self, keypoint, attributes):
        """
        Convert a keypoint's attribute dictionary
        into a readable sentence.
        """

        keypoint_attributes = []

        for attribute in attributes:

            for key, value in attribute.items():

                if isinstance(value, list):
                    selected = value[0]
                else:
                    selected = value

                keypoint_attributes.append(
                    f"{selected} {key}"
                )

        return (
            ", ".join(keypoint_attributes)
            + f" {keypoint}"
        )

    def drop_random_keypoints(self, keypoints_attributes_dict, drop_parts):

        updated_dict = keypoints_attributes_dict.copy()
        num_drop = min(drop_parts, len(updated_dict))

        if num_drop > 0:
            drop_keypoints = random.sample(list(updated_dict.keys()),num_drop)
            for keypoint in drop_keypoints:
                del updated_dict[keypoint]

        return updated_dict


    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """
        Load annotation metadata.
        """

        image_file = os.path.join(
            self.img_prefix,
            self.id2name[img_id]
        )

        ann_ids = self.coco.getAnnIds(
            imgIds=img_id,
            iscrowd=False
        )

        objs = self.coco.loadAnns(ann_ids)

        rec = []

        for obj in objs:

            category_info = self.categories[obj["category_id"]]
            keypoints_attributes_dict = (category_info.get("keypoint_attributes_by_category",{}))

            keypoints_attributes_dict = self.drop_random_keypoints(keypoints_attributes_dict, self.drop_parts)


            attributes_for_keypoints = []

            for keypoint, attributes in (keypoints_attributes_dict.items()):


                attributes_for_a_keypoint = (self.get_attributes_for_a_keypoint(keypoint, attributes))
                attributes_for_keypoints.append(attributes_for_a_keypoint)

            attributes_for_keypoints = "; ".join(attributes_for_keypoints)

            if attributes_for_keypoints: attributes_for_keypoints += "."


            category_name = ('"' + category_info["name"] + '"')
            caption = (f"It is a species of {category_name} " f"as it has {attributes_for_keypoints}")
            caption += " <|endoftext|>"

            prefix_str = "It is a species of "
            category_text = category_name
            suffix_text = f" as it has {attributes_for_keypoints}"
            
            # 2. Calculate category start and end token indices
            start_idx = len(self.tokenizer.encode(prefix_str, add_special_tokens=False))
            end_idx = len(self.tokenizer.encode(prefix_str + category_name, add_special_tokens=False))
            
            # 3. Mask using a simple slice
            category_mask = torch.zeros(self.args.max_seq_len, dtype=torch.bool)
            category_mask[start_idx:end_idx] = True
            
            target_enc = self.tokenizer(caption, padding="max_length", truncation=True,
                        max_length=self.args.max_seq_len, return_tensors="pt")

            target_ids = (target_enc.input_ids.squeeze(0))
            target_mask = (target_enc.attention_mask.squeeze(0))

            rec.append({

                "image_file": image_file,

                "prompt_ids": self.prompt_ids,
                "prompt_mask": self.prompt_mask,
                
                "target_ids": target_ids,
                "target_mask": target_mask,
                "category_name": category_name.strip('"'),
                "caption": caption,
                "category_mask":category_mask
            })

        return rec


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
        }
