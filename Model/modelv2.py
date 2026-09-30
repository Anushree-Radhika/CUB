import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.VisionEncoderPooled import VisionEncoder
from Model.Decoder import GPT2Decoder
from Model.Qwen3Decoder import Qwen3Decoder

import os


import matplotlib.pyplot as plt
def visualize_and_save_similarity_heatmap(original_image_tensor, similarity_matrix, prompt_string, save_path):
    """
    Enhanced Heatmap Generator with Anisotropy Correction & Centering.
    Isolates true semantic matches without retraining.
    """
    # 1. Isolate prompt matrix: (num_patches,)
    similarity_map = similarity_matrix[0, 0] 
    
    num_patches = similarity_map.size(0)
    grid_size = int(num_patches ** 0.5)
    
    # ------------------------------------------------------------------
    # FIX 1: Subtract the Mean (Center the similarities)
    # This removes the GPT-2 baseline anisotropy noise (e.g. baseline 0.70)
    # ------------------------------------------------------------------
    mean_sim = similarity_map.mean()
    std_sim = similarity_map.std() + 1e-8
    
    # Standardize scores (Z-score normalization over patches)
    norm_sim_map = (similarity_map - mean_sim) / std_sim
    
    # Apply ReLU to discard below-average matching patches (suppress background)
    norm_sim_map = F.relu(norm_sim_map)
    
    # Reshape to 2D grid: (grid_size, grid_size)
    norm_sim_map = norm_sim_map.view(grid_size, grid_size)

    # ------------------------------------------------------------------
    # FIX 2: Bilinear Rescaling to Image Dimensions
    # ------------------------------------------------------------------
    B, C, H, W = original_image_tensor.shape
    scaled_sim_map = norm_sim_map.unsqueeze(0).unsqueeze(0) # (1, 1, grid, grid)
    
    heatmap_resized = F.interpolate(scaled_sim_map, size=(H, W), mode='bilinear', align_corners=False)
    heatmap_np = heatmap_resized.squeeze().cpu().numpy()

    # Min-max scale the final heatmap for display (0.0 to 1.0)
    if heatmap_np.max() > 0:
        heatmap_np = heatmap_np / heatmap_np.max()

    # ------------------------------------------------------------------
    # FIX 3: Plot with Matplotlib Overlay
    # ------------------------------------------------------------------
    img_tensor = original_image_tensor[0].cpu().detach()
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    img_np = img_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)
    
    # Overlay using jet colormap with thresholded alpha
    heatmap_overlay = ax.imshow(heatmap_np, cmap='jet', alpha=0.55)
    
    plt.title(f"Targeting: '{prompt_string}'", fontsize=12, pad=10)
    plt.axis('off')
    fig.colorbar(heatmap_overlay, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"[SUCCESS] Centered Heatmap saved: {save_path}")
def visualize_and_save_similarity_heatmap_old(original_image_tensor, similarity_matrix, prompt_string, save_path):
    """
    Visualizes Cosine Similarity using ONLY PyTorch and Matplotlib.
    Bypasses SciPy, Sklearn, and OpenCV completely to prevent NumPy 2.0 crashes.
    """
    # 1. Select the first batch item and first prompt similarity
    # similarity_matrix shape: (B, num_prompts, num_patches)
    similarity_map = similarity_matrix[0, 0] # (num_patches,)
    
    # Calculate grid size (e.g., 196 patches -> 14x14 grid)
    num_patches = similarity_map.size(0)
    grid_size = int(num_patches ** 0.5)
    
    # Reshape to 2D grid: (grid_size, grid_size)
    similarity_map = similarity_map.view(grid_size, grid_size)

    # Min-max normalize similarity values to 0.0 - 1.0 range
    min_sim, max_sim = similarity_map.min(), similarity_map.max()
    norm_sim_map = (similarity_map - min_sim) / (max_sim - min_sim + 1e-8)

    # 2. Rescale heatmap to match input image dimensions using PyTorch
    B, C, H, W = original_image_tensor.shape
    scaled_sim_map = norm_sim_map.unsqueeze(0).unsqueeze(0) # (1, 1, grid, grid)
    
    # Bilinear interpolation up to full image height & width
    heatmap_resized = F.interpolate(scaled_sim_map, size=(H, W), mode='bilinear', align_corners=False)
    heatmap_np = heatmap_resized.squeeze().cpu().numpy()

    # 3. Convert PyTorch image tensor to NumPy format for plotting
    img_tensor = original_image_tensor[0].cpu().detach()
    # Normalize tensor channels to 0-1 for plotting if they aren't already
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    img_np = img_tensor.permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)

    # 4. Plot original image + heatmap overlay using Matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Render base image
    ax.imshow(img_np)
    
    # Overlay heat map using 'jet' colormap with 50% transparency (alpha=0.5)
    heatmap_overlay = ax.imshow(heatmap_np, cmap='jet', alpha=0.5)
    
    # Add title and colorbar legend
    plt.title(f"Patch Similarity: '{prompt_string}'", fontsize=12, pad=10)
    plt.axis('off')
    fig.colorbar(heatmap_overlay, ax=ax, fraction=0.046, pad=0.04)

    # Save output image
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"[SUCCESS] Heatmap saved without SciPy/OpenCV to: {save_path}")

def visualize_and_save_similarity_heatmap(original_image_tensor, similarity_matrix, prompt_string, save_path):
    # 1. Input Check: Ensure we have a matching BATCH_SIZE
    if original_image_tensor.size(0) != similarity_matrix.size(0):
        raise ValueError(f"Batch size mismatch. Got Images={original_image_tensor.size(0)}, Sims={similarity_matrix.size(0)}")
    
    # 2. Reshape and Normalize the similarity matrix
    # similarity_matrix is (B, num_prompts, num_patches).
    # We select the FIRST prompt (index 0) and the FIRST image in the batch (index 0).
    similarity_map = similarity_matrix[0, 0] # (num_patches,)
    
    # Calculate grid size (e.g., sqrt(196) = 14)
    num_patches = similarity_map.size(0)
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Cosine similarity map (size {num_patches}) must be a square number.")

    # Reshape to (H, W) for interpolation
    similarity_map = similarity_map.view(grid_size, grid_size)

    # Convert similarity map to a normalized 0.0 to 1.0 float tensor
    min_sim = similarity_map.min()
    max_sim = similarity_map.max()
    norm_sim_map = (similarity_map - min_sim) / (max_sim - min_sim)

    # 3. Scale and Resize the Heatmap to match the image dimensions
    B, C, H, W = original_image_tensor.shape
    scaled_sim_map = norm_sim_map.unsqueeze(0).unsqueeze(0) # (1, 1, grid, grid)
    resized_heatmap_tensor = F.interpolate(scaled_sim_map, size=(H, W), mode='bilinear', align_corners=False)
    
    # Convert to Numpy (H, W) array, scaled to 0-255 range for OpenCV
    resized_heatmap_np = (resized_heatmap_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # 4. Generate the Visualized Blended Image
    # Convert original tensor back to RGB Numpy image (B, C, H, W) -> (H, W, C)
    original_img_np = original_image_tensor[0].permute(1, 2, 0).cpu().numpy()
    
    # Normalize image pixel range and convert to BGR (required by OpenCV)
    original_img_np = ((original_img_np - original_img_np.min()) / (original_img_np.max() - original_img_np.min()) * 255).astype(np.uint8)
    original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

    # Create the COLOR heatmap (Map 0-255 to JET colormap: low similarity = blue, high similarity = red)
    heatmap_colored = cv2.applyColorMap(resized_heatmap_np, cv2.COLORMAP_JET)

    # 5. SUPERIMPOSE! (Overlay Heatmap onto the original image)
    # The weight parameters (alpha and beta) control the transparency of the overlay
    alpha = 0.6  # Weight of the heatmap (0.0 to 1.0, higher means darker/more dominant heatmap)
    beta = 1.0 - alpha
    blended_image = cv2.addWeighted(heatmap_colored, alpha, original_img_bgr, beta, 0)

    # Optional: Plot and Save the final output image with title
    # (We save the image directly with OpenCV to avoid Matplotlib dependency if preferred)
    cv2.imwrite(save_path, blended_image)
    
    print(f"[INFO] Successfully generated similarity visualization.")
    print(f"       Prompt used: '{prompt_string}'")
    print(f"       Output saved to: '{save_path}'")


class TraitGen(nn.Module):
    """
    TraitGen: Image Captioning Model

    Components:
        - VisionEncoder: extracts image representations.
        - Bridge: projects visual features into GPT-2 embedding space.
        - GPT2Decoder: generates captions conditioned on image features.
    """

    def __init__(self, args, vision_encoder=None):
        super().__init__()

        self.args = args
        self.vision_encoder = VisionEncoder(args) if vision_encoder is None else vision_encoder
        
        if(args.decoder_model == "openai-community/gpt2-medium"):
            self.decoder = GPT2Decoder(args)
        elif(args.decoder_model == "Qwen/Qwen3-4B-Base" or args.decoder_model == "Qwen/Qwen3-4B"):
            self.decoder = Qwen3Decoder(args)
        else:
            # fallback to GPT2
            self.decoder = GPT2Decoder(args)
            print("Fallback to GPT2 decoder")
            print("="*80)

        self.bridge = Bridge(vision_dim=args.encoder_op_dim, hidden_dim=self.decoder.hidden_dim)

    # ============================================================
    # Prepare inputs for GPT-2 training/validation
    # ============================================================
    def input2decoder(self, prompt_ids, prompt_mask, prefix_embeds, target_ids, target_mask):

        prompt_embeds = self.decoder.llm.get_input_embeddings()(prompt_ids)
        target_embeds = self.decoder.llm.get_input_embeddings()(target_ids)

        inputs_embeds = torch.cat([prompt_embeds, prefix_embeds, target_embeds], dim=1)

        B, PREFIX_LEN = prompt_ids.size(0), prefix_embeds.size(1)
        device = prompt_ids.device

        prompt_labels = torch.full(prompt_ids.shape, -100, device=device, dtype=torch.long)
        prefix_labels = torch.full((B, PREFIX_LEN), -100, device=device, dtype=torch.long)
        #print(prompt_ids.shape)
        #print(PREFIX_LEN)
        #print(target_mask.shape)
        labels = torch.cat([prompt_labels, prefix_labels, target_ids], dim=1)

        prefix_mask = torch.ones((B, PREFIX_LEN), device=device, dtype=torch.long)
        # category_mask = torch.cat([prompt_mask,prefix_mask,cat_mask],dim=1).to(torch.long)
        full_mask = torch.cat([prompt_mask, prefix_mask, target_mask], dim=1)
        
        return inputs_embeds, full_mask, labels


    # ============================================================
    # Training / validation forward pass
    # ============================================================
    def forward(self, image, prompt_ids, prompt_mask, target_ids, target_mask, category_mask):

        image_features = self.vision_encoder(image).permute(0, 2, 1)
        prefix_embeds = self.bridge(image_features)

        inputs_embeds, attention_mask, labels = self.input2decoder(
            prompt_ids, prompt_mask, prefix_embeds, target_ids, target_mask)
        batch_sz,num_pref_tok,_ = prefix_embeds.shape

        # prefix_mask = torch.ones((batch_sz,num_pref_tok),dtype=torch.long,device=prefix_embeds.device)
        
        image_len = prefix_embeds.size(1)
        prompt_len = prompt_ids.size(1)
        
        inputs_embeds = inputs_embeds.to(torch.float16)
        
        outputs,loss = self.decoder(inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, 
                labels=labels, 
                species_mask=category_mask, 
                image_len=image_len,
                prompt_len=prompt_len)
        # loss = outputs.loss

        return loss

    # ============================================================
    # Caption generation
    # ============================================================

    @torch.no_grad()
    def generate_caption(self, image, prompt_ids, prompt_mask):

        image_features = self.vision_encoder(image).permute(0, 2, 1)
        prefix_embeds = self.bridge(image_features)

        prompt_embeds = self.decoder.gpt2.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([prompt_embeds, prefix_embeds], dim=1)

        B, PREFIX_LEN = prompt_ids.size(0), prefix_embeds.size(1)
        prefix_mask = torch.ones((B, PREFIX_LEN), device=prompt_ids.device, dtype=torch.long)
        attention_mask = torch.cat([prompt_mask, prefix_mask], dim=1)

        generated = self.decoder.gpt2.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.92,
            repetition_penalty=1.2, eos_token_id=self.decoder.tokenizer.eos_token_id,
            pad_token_id=self.decoder.tokenizer.eos_token_id,)

        generated_text = self.decoder.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return generated_text
    
    @torch.no_grad()
    def generate_image_patches_old(self,image,prompt_list,device="cuda",base_output_path="./similarity_analysis"):
        image_features = self.vision_encoder(image).permute(0, 2, 1)
        prefix_embeds = self.bridge(image_features)
        
        # Step 2: Unfold raw input pixels directly into flattened patch representations (Zero extra weights)
        B, C, H, W = image.shape
        num_patches = prefix_embeds.size(1)
        patch_size = int((H * W / num_patches) ** 0.5)

        reconstructed_patches = image.unfold(2, patch_size, patch_size)\
                                    .unfold(3, patch_size, patch_size)\
                                    .permute(0, 2, 3, 1, 4, 5)\
                                    .contiguous()\
                                    .view(B, num_patches, -1)

        # Step 3: Tokenize text prompts and extract embeddings from pre-trained GPT-2
        tokenized_inputs = self.decoder.tokenizer(
            prompt_list, 
            padding=True, 
            return_tensors="pt"
        ).to(device)

        # Retrieve text token embeddings from existing LLM layer
        # Shape: (num_prompts, seq_len, hidden_dim)
        text_token_embeds = self.decoder.gpt2.get_input_embeddings()(tokenized_inputs.input_ids)

        # Masked average to pool token vectors into a single prompt vector
        attention_mask = tokenized_inputs.attention_mask.unsqueeze(-1)  # (num_prompts, seq_len, 1)
        text_embeds = (text_token_embeds * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        # Compute cosine similarity between text prompts and image patches
        prefix_embeds_norm = F.normalize(prefix_embeds, p=2, dim=-1)  # (B, num_patches, hidden_dim)
        text_embeds_norm = F.normalize(text_embeds, p=2, dim=-1)        # (num_prompts, hidden_dim)

        # Similarity matrix: (B, num_prompts, num_patches)
        similarity_matrix = torch.matmul(text_embeds_norm, prefix_embeds_norm.transpose(-1, -2))
        
        output_dict = {
            "patch_embeddings": prefix_embeds,
            "reconstructed_patches": reconstructed_patches,
            "similarity_matrix": similarity_matrix
        }
        # --- End existing generate_image_patches logic ---


        # ============================================================
        # STEP 4: VISUALIZE SIMILARITYHEATMAP
        # ============================================================
        if not os.path.exists(base_output_path):
            os.makedirs(base_output_path)

        # Iterate through all prompts and generate a visualization for each one
        # Note: original_image_tensor must retain its full spatial dimension (B, C, H, W)
        for p_idx, current_prompt_string in enumerate(prompt_list):
            # Slice the similarity matrix to isolate the matrix for just this prompt: (B, 1, num_patches)
            specific_prompt_sim_matrix = similarity_matrix[:, p_idx:p_idx+1, :]
            
            # Ensure filenames are safe for saving (sanitize/remove spaces if needed)
            filename_prompt = current_prompt_string.replace(' ', '_').replace('"', '')[:30] # Limit filename length
            full_save_path = os.path.join(base_output_path, f"vis_sim_matrix_P{p_idx}_{filename_prompt}.jpg")

            # CALL THE STANDALONE VISUALIZATION FUNCTION from Part 1
            # It handles all interpolation, color-mapping, blending, and saving.
            visualize_and_save_similarity_heatmap(image, specific_prompt_sim_matrix, current_prompt_string, full_save_path)

        return output_dict
    @torch.no_grad()
    def generate_image_patches(self, image, prompt_list,device="cuda",base_output_path="./similarity_analysis"):
        """
        Supports complex multi-word prompts by using GPT-2's contextualized 
        transformer representations rather than static mean-pooled input embeddings.
        """
        device = image.device

        # 1. Vision Features -> Bridge
        image_features = self.vision_encoder(image).permute(0, 2, 1) # (B, num_patches, vision_dim)
        prefix_embeds = self.bridge(image_features)                  # (B, num_patches, hidden_dim)

        # 2. Tokenize Prompts
        tokenized_inputs = self.decoder.tokenizer(
            prompt_list, 
            padding=True, 
            return_tensors="pt"
        ).to(device)

        input_ids = tokenized_inputs.input_ids
        attention_mask = tokenized_inputs.attention_mask

        # ----------------------------------------------------------------------
        # FIX FOR MULTI-WORD PROMPTS: Contextual Transformer Pass
        # Pass input IDs through GPT-2's trunk to get fully contextualized tokens
        # ----------------------------------------------------------------------
        gpt2_outputs = self.decoder.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = gpt2_outputs.last_hidden_state  # (num_prompts, seq_len, hidden_dim)

        # Extract the LAST non-padded token position for each prompt in the batch
        # (This vector contains the causal summary of the entire multi-word phrase)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=device)
        text_embeds = hidden_states[batch_indices, sequence_lengths] # (num_prompts, hidden_dim)

        # 3. Normalized Cosine Similarity
        prefix_embeds_norm = F.normalize(prefix_embeds, p=2, dim=-1) # (B, num_patches, hidden_dim)
        text_embeds_norm = F.normalize(text_embeds, p=2, dim=-1)     # (num_prompts, hidden_dim)

        # Similarity matrix: (B, num_prompts, num_patches)
        similarity_matrix = torch.matmul(text_embeds_norm, prefix_embeds_norm.transpose(-1, -2))

        output_dict = {
            "patch_embeddings": prefix_embeds,
            "similarity_matrix": similarity_matrix
        }
        
        if not os.path.exists(base_output_path):
                    os.makedirs(base_output_path)
        
        # Iterate through all prompts and generate a visualization for each one
        # Note: original_image_tensor must retain its full spatial dimension (B, C, H, W)
        for p_idx, current_prompt_string in enumerate(prompt_list):
            # Slice the similarity matrix to isolate the matrix for just this prompt: (B, 1, num_patches)
            specific_prompt_sim_matrix = similarity_matrix[:, p_idx:p_idx+1, :]
                    
            # Ensure filenames are safe for saving (sanitize/remove spaces if needed)
            filename_prompt = current_prompt_string.replace(' ', '_').replace('"', '')[:30] # Limit filename length
            full_save_path = os.path.join(base_output_path, f"vis_sim_matrix_P{p_idx}_{filename_prompt}.jpg")
        
            # CALL THE STANDALONE VISUALIZATION FUNCTION from Part 1
            # It handles all interpolation, color-mapping, blending, and saving.
            visualize_and_save_similarity_heatmap(image, specific_prompt_sim_matrix, current_prompt_string, full_save_path)
        
        return output_dict
    @torch.no_grad()
    def visualize_trait_attention_old(self, image, trait_text, save_path="./attention_map.png"):
        """
        Extracts GPT-2's RAW internal cross-attention weights over the image patches
        for a specific generated trait.
        """
        self.eval()
        device = image.device

        # 1. Forward Vision Encoder & Bridge
        image_features = self.vision_encoder(image)
        if image_features.dim() == 3 and image_features.size(1) == self.args.encoder_op_dim:
            image_features = image_features.permute(0, 2, 1)

        prefix_embeds = self.bridge(image_features)  # (1, num_patches, hidden_dim)

        # 2. Tokenize the trait
        tokenizer = self.decoder.tokenizer
        target_ids = tokenizer(trait_text, return_tensors="pt").input_ids.to(device)
        target_mask = torch.ones_like(target_ids)

        prompt_ids = torch.empty((1, 0), dtype=torch.long, device=device)
        prompt_mask = torch.empty((1, 0), dtype=torch.long, device=device)

        # 3. Prepare inputs for GPT-2
        inputs_embeds, attention_mask, _ = self.input2decoder(
            prompt_ids, prompt_mask, prefix_embeds, target_ids, target_mask
        )

        # 4. Pass through GPT-2 requesting raw attention outputs directly from the trunk.
        # Force eager attention for this call so per-head weights are actually returned
        # (sdpa/flash-attn backends won't materialize them).
        gpt2_base = getattr(self.decoder.gpt2, "transformer", self.decoder.gpt2)
        prev_attn_impl = getattr(gpt2_base.config, "_attn_implementation", None)
        if hasattr(gpt2_base.config, "_attn_implementation"):
            gpt2_base.config._attn_implementation = "eager"

        try:
            outputs = gpt2_base(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
        finally:
            if prev_attn_impl is not None:
                gpt2_base.config._attn_implementation = prev_attn_impl

        if outputs.attentions is None:
            raise ValueError(
                "GPT-2 returned None for attentions. Ensure the model config allows output_attentions=True."
            )

        # outputs.attentions: tuple of (num_layers,) tensors, each (batch, num_heads, seq_len, seq_len)
        num_patches = prefix_embeds.size(1)
        text_len = target_ids.size(1)

        # Stack layers -> (num_layers, batch, num_heads, seq_len, seq_len); squeeze batch (batch_size=1)
        all_attentions = torch.stack(outputs.attentions).squeeze(1)

        # Attention flowing FROM text tokens (last text_len query positions, since sequence is
        # [prefix_patches, target_tokens]) TO patch tokens (first num_patches key positions).
        # Average across all layers and heads.
        patch_attention = all_attentions[:, :, -text_len:, :num_patches].mean(dim=(0, 1, 2))  # (num_patches,)

        # 5. Reshape to a square grid — validate rather than silently truncating.
        grid_size = int(round(num_patches ** 0.5))
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"num_patches ({num_patches}) is not a perfect square; "
                f"cannot reshape into a square attention grid."
            )
        attn_grid = patch_attention.view(grid_size, grid_size)

        # Center + standardize to strip GPT-2's attention baseline, then clip negatives
        # so only above-average patches light up (same trick used in the similarity heatmap).
        mean_attn = attn_grid.mean()
        std_attn = attn_grid.std() + 1e-8
        attn_grid = F.relu((attn_grid - mean_attn) / std_attn)

        # 6. Resize to original image dimensions
        B, C, H, W = image.shape
        heatmap_resized = F.interpolate(
            attn_grid.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()

        # 7. Plot original image + Attention overlay
        img_tensor = image[0].cpu().detach()
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
        img_np = img_tensor.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        overlay = ax.imshow(heatmap_resized, cmap='jet', alpha=0.60)

        plt.title(f"GPT-2 Cross-Attention Map for Trait:\n'{trait_text}'", fontsize=11, pad=10)
        plt.axis('off')
        fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"[SUCCESS] Attention map saved to: {save_path}")
    @torch.no_grad()
    def _get_patch_attention(self, prefix_embeds, text, device):
        """
        Runs GPT-2 on [image_patches, text_tokens] and returns the raw
        (unnormalized) mean attention from text tokens -> patch tokens.
        Shape: (num_patches,)
        """
        tokenizer = self.decoder.tokenizer
        target_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        target_mask = torch.ones_like(target_ids)

        prompt_ids = torch.empty((1, 0), dtype=torch.long, device=device)
        prompt_mask = torch.empty((1, 0), dtype=torch.long, device=device)

        inputs_embeds, attention_mask, _ = self.input2decoder(
            prompt_ids, prompt_mask, prefix_embeds, target_ids, target_mask
        )

        gpt2_base = getattr(self.decoder.gpt2, "transformer", self.decoder.gpt2)
        prev_attn_impl = getattr(gpt2_base.config, "_attn_implementation", None)
        if hasattr(gpt2_base.config, "_attn_implementation"):
            gpt2_base.config._attn_implementation = "eager"

        try:
            outputs = gpt2_base(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
        finally:
            if prev_attn_impl is not None:
                gpt2_base.config._attn_implementation = prev_attn_impl

        if outputs.attentions is None:
            raise ValueError(
                "GPT-2 returned None for attentions. Ensure the model config allows output_attentions=True."
            )

        num_patches = prefix_embeds.size(1)
        text_len = target_ids.size(1)

        # (num_layers, batch, num_heads, seq_len, seq_len) -> squeeze batch (batch_size=1)
        all_attentions = torch.stack(outputs.attentions).squeeze(1)

        # Use only the LAST token's attention (most contextualized) rather than
        # averaging over every target token, which dilutes signal with
        # sink-prone leading tokens (BOS, articles, etc.)
        patch_attention = all_attentions[:, :, -1, :num_patches].mean(dim=(0, 1))  # (num_patches,)

        return patch_attention


    @torch.no_grad()
    def visualize_trait_attention_older(self, image, trait_text, save_path="./attention_map.png",
                                baseline_text="a photo"):
        """
        Extracts GPT-2's internal cross-attention over image patches for a specific
        trait, with the static positional/sink bias removed via contrastive
        baseline subtraction: attn(trait_text) - attn(baseline_text).

        baseline_text: a semantically neutral prompt run through the same model.
        Its attention pattern captures the generic positional/sink bias that
        shows up for any input; subtracting it isolates what's specific to
        trait_text.
        """
        self.eval()
        device = image.device

        # 1. Forward Vision Encoder & Bridge
        image_features = self.vision_encoder(image)
        if image_features.dim() == 3 and image_features.size(1) == self.args.encoder_op_dim:
            image_features = image_features.permute(0, 2, 1)

        prefix_embeds = self.bridge(image_features)  # (1, num_patches, hidden_dim)
        num_patches = prefix_embeds.size(1)

        # 2. Raw attention for the trait and for the neutral baseline
        trait_attn = self._get_patch_attention(prefix_embeds, trait_text, device)
        baseline_attn = self._get_patch_attention(prefix_embeds, baseline_text, device)

        # 3. Contrastive subtraction: cancel out the shared positional/sink bias,
        # keep only what's specific to trait_text. Clip negatives (patches the
        # baseline attended to MORE than the trait) so the map highlights only
        # trait-preferring patches.
        diff_attn = F.relu(trait_attn - baseline_attn)

        # 4. Reshape to a square grid — validate rather than silently truncating.
        grid_size = int(round(num_patches ** 0.5))
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"num_patches ({num_patches}) is not a perfect square; "
                f"cannot reshape into a square attention grid."
            )
        attn_grid = diff_attn.view(grid_size, grid_size)

        # Z-score + ReLU for contrast (same trick as the similarity heatmap fn)
        mean_attn = attn_grid.mean()
        std_attn = attn_grid.std() + 1e-8
        attn_grid = F.relu((attn_grid - mean_attn) / std_attn)

        # 5. Resize to original image dimensions
        B, C, H, W = image.shape
        heatmap_resized = F.interpolate(
            attn_grid.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()

        # 6. Plot original image + Attention overlay
        img_tensor = image[0].cpu().detach()
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
        img_np = img_tensor.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        overlay = ax.imshow(heatmap_resized, cmap='jet', alpha=0.60)

        plt.title(f"GPT-2 Cross-Attention (contrastive) for Trait:\n'{trait_text}'"
                f"\n(baseline: '{baseline_text}')", fontsize=10, pad=10)
        plt.axis('off')
        fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"[SUCCESS] Contrastive attention map saved to: {save_path}")
    @torch.no_grad()
    def visualize_trait_attention(self, image, trait_text, save_path="./attention_map.png",
                                baseline_text="species identification and corresponding textual explanation task.", sink_patches=(0,)):
        """
        Contrastive attention map with:
        - baseline text length-matched to trait text (removes length-driven
            sink-magnitude confound)
        - known sink patch indices excluded from the map entirely
        """
        self.eval()
        device = image.device

        image_features = self.vision_encoder(image)
        if image_features.dim() == 3 and image_features.size(1) == self.args.encoder_op_dim:
            image_features = image_features.permute(0, 2, 1)
        prefix_embeds = self.bridge(image_features)
        num_patches = prefix_embeds.size(1)

        tokenizer = self.decoder.tokenizer

        # --- length-match baseline to trait ---
        trait_ids = tokenizer(trait_text, return_tensors="pt").input_ids
        trait_len = trait_ids.size(1)
        base_ids = tokenizer(baseline_text, return_tensors="pt").input_ids
        if base_ids.size(1) < trait_len:
            pad_id = tokenizer.eos_token_id
            pad = torch.full((1, trait_len - base_ids.size(1)), pad_id, dtype=torch.long)
            base_ids = torch.cat([base_ids, pad], dim=1)
        elif base_ids.size(1) > trait_len:
            base_ids = base_ids[:, :trait_len]
        baseline_text_matched = tokenizer.decode(base_ids[0])

        trait_attn = self._get_patch_attention(prefix_embeds, trait_text, device)
        baseline_attn = self._get_patch_attention(prefix_embeds, baseline_text_matched, device)

        diff_attn = trait_attn - baseline_attn

        # --- exclude known sink patches before any normalization ---
        valid_mask = torch.ones(num_patches, dtype=torch.bool, device=diff_attn.device)
        for idx in sink_patches:
            valid_mask[idx] = False
        diff_attn = diff_attn.clone()
        diff_attn[~valid_mask] = diff_attn[valid_mask].mean()  # neutralize, don't zero (avoids fake "cold" spot)

        diff_attn = F.relu(diff_attn)

        grid_size = int(round(num_patches ** 0.5))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"num_patches ({num_patches}) is not a perfect square.")
        attn_grid = diff_attn.view(grid_size, grid_size)

        mean_attn = attn_grid.mean()
        std_attn = attn_grid.std() + 1e-8
        attn_grid = F.relu((attn_grid - mean_attn) / std_attn)

        B, C, H, W = image.shape
        heatmap_resized = F.interpolate(
            attn_grid.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()

        img_tensor = image[0].cpu().detach()
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
        img_np = img_tensor.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        overlay = ax.imshow(heatmap_resized, cmap='jet', alpha=0.60)
        plt.title(f"GPT-2 Cross-Attention (contrastive, sink-masked) for Trait:\n'{trait_text}'", fontsize=10, pad=10)
        plt.axis('off')
        fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"[SUCCESS] Sink-masked attention map saved to: {save_path}")
        
    def visualize_trait_attribution(self, image, trait_text, save_path="./attribution_map.png"):
        """
        Gradient-based attribution: measures how much each image patch actually
        influenced GPT-2's probability of generating trait_text, rather than
        how much attention weight it received (which is a weaker, indirect proxy).
        """
        self.eval()
        device = image.device

        # 1. Vision features -> Bridge. Keep grad this time.
        image_features = self.vision_encoder(image)
        if image_features.dim() == 3 and image_features.size(1) == self.args.encoder_op_dim:
            image_features = image_features.permute(0, 2, 1)

        prefix_embeds = self.bridge(image_features)  # (1, num_patches, hidden_dim)
        prefix_embeds.retain_grad()
        prefix_embeds.requires_grad_(True)

        # 2. Tokenize the trait text
        tokenizer = self.decoder.tokenizer
        target_ids = tokenizer(trait_text, return_tensors="pt").input_ids.to(device)

        prompt_ids = torch.empty((1, 0), dtype=torch.long, device=device)
        prompt_embeds = self.decoder.gpt2.get_input_embeddings()(prompt_ids)
        target_embeds = self.decoder.gpt2.get_input_embeddings()(target_ids)

        inputs_embeds = torch.cat([prompt_embeds, prefix_embeds, target_embeds], dim=1)

        prefix_len = prefix_embeds.size(1)
        target_len = target_ids.size(1)
        attention_mask = torch.ones((1, prefix_len + target_len), device=device, dtype=torch.long)

        # 3. Forward pass, get logits
        outputs = self.decoder.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # 4. For each target token, get the log-prob GPT-2 assigned to the CORRECT
        # next token, at the position right before it. Sum them — this is the
        # total log-likelihood GPT-2 gave to generating "trait_text" from the prefix.
        # Position of the first target token's prediction is prefix_len - 1 (0-indexed).
        log_probs = F.log_softmax(logits, dim=-1)
        total_log_prob = 0.0
        for t in range(target_len):
            pred_pos = prefix_len - 1 + t
            token_id = target_ids[0, t]
            total_log_prob = total_log_prob + log_probs[0, pred_pos, token_id]

        # 5. Backprop to get gradient of that log-prob w.r.t. each patch embedding
        self.decoder.gpt2.zero_grad()
        total_log_prob.backward()

        grad = prefix_embeds.grad[0]  # (num_patches, hidden_dim)

        # 6. Gradient x Input (a standard, sharper attribution than raw gradient norm)
        attribution = (grad * prefix_embeds[0].detach()).sum(dim=-1)  # (num_patches,)
        attribution = attribution.detach()

        # Center + suppress negative-influence patches (or keep them if you want
        # to see what pushed AWAY from this word — see note below)
        attribution = F.relu(attribution)

        num_patches = prefix_embeds.size(1)
        grid_size = int(round(num_patches ** 0.5))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"num_patches ({num_patches}) is not a perfect square.")
        attr_grid = attribution.view(grid_size, grid_size)

        mean_a, std_a = attr_grid.mean(), attr_grid.std() + 1e-8
        attr_grid = F.relu((attr_grid - mean_a) / std_a)

        B, C, H, W = image.shape
        heatmap_resized = F.interpolate(
            attr_grid.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()

        img_tensor = image[0].cpu().detach()
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
        img_np = img_tensor.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        overlay = ax.imshow(heatmap_resized, cmap='jet', alpha=0.60)
        plt.title(f"Patch Attribution (Grad×Input) for Trait:\n'{trait_text}'", fontsize=11, pad=10)
        plt.axis('off')
        fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"[SUCCESS] Attribution map saved to: {save_path}")
    def visualize_trait_integrated_gradients(self, image, trait_text, save_path="./ig_attribution_map.png",
                                            steps=20, baseline_mode="mean"):
        """
        Integrated Gradients attribution map: measures each patch's contribution
        to GPT-2's log-probability of generating trait_text, integrated along a
        path from a baseline (blank) patch embedding to the actual one. More
        robust to texture/edge-driven noise than single-point Grad x Input.
        """
        self.eval()
        device = image.device

        # 1. Vision features -> Bridge (no grad needed here, we build interp separately)
        image_features = self.vision_encoder(image)
        if image_features.dim() == 3 and image_features.size(1) == self.args.encoder_op_dim:
            image_features = image_features.permute(0, 2, 1)
        prefix_embeds = self.bridge(image_features).detach()  # (1, num_patches, hidden_dim)
        num_patches = prefix_embeds.size(1)

        # 2. Baseline prefix embedding to integrate FROM.
        # "zero": a blank/uninformative reference (standard IG choice).
        # "mean": each patch replaced by the mean of all patches (a "flat, generic
        #         patch" reference) -- sometimes less biased toward the model's
        #         behavior on literally-zero inputs, which it may never see in training.
        if baseline_mode == "zero":
            baseline_prefix = torch.zeros_like(prefix_embeds)
        elif baseline_mode == "mean":
            baseline_prefix = prefix_embeds.mean(dim=1, keepdim=True).expand_as(prefix_embeds)
        else:
            raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

        # 3. Tokenize the trait
        tokenizer = self.decoder.tokenizer
        target_ids = tokenizer(trait_text, return_tensors="pt").input_ids.to(device)
        target_len = target_ids.size(1)

        prompt_ids = torch.empty((1, 0), dtype=torch.long, device=device)
        prompt_embeds = self.decoder.gpt2.get_input_embeddings()(prompt_ids)
        target_embeds = self.decoder.gpt2.get_input_embeddings()(target_ids)

        # 4. Integrate gradients along the path baseline -> prefix_embeds
        total_grad = torch.zeros_like(prefix_embeds)
        alphas = torch.linspace(0.0, 1.0, steps, device=device)

        for alpha in alphas:
            interp = baseline_prefix + alpha * (prefix_embeds - baseline_prefix)
            interp = interp.clone().requires_grad_(True)

            inputs_embeds = torch.cat([prompt_embeds, interp, target_embeds], dim=1)
            prefix_len = interp.size(1)
            attention_mask = torch.ones((1, prefix_len + target_len), device=device, dtype=torch.long)

            outputs = self.decoder.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            log_probs = F.log_softmax(outputs.logits, dim=-1)

            total_log_prob = 0.0
            for t in range(target_len):
                pred_pos = prefix_len - 1 + t
                total_log_prob = total_log_prob + log_probs[0, pred_pos, target_ids[0, t]]

            self.decoder.gpt2.zero_grad()
            grad = torch.autograd.grad(total_log_prob, interp, retain_graph=False)[0]
            total_grad = total_grad + grad.detach()

        avg_grad = total_grad / steps

        # 5. IG attribution = (prefix - baseline) * avg_grad, summed over hidden dim
        attribution = ((prefix_embeds - baseline_prefix) * avg_grad).sum(dim=-1)[0]  # (num_patches,)
        attribution = F.relu(attribution)  # keep only patches that pushed TOWARD the trait word

        # 6. Reshape to square grid
        grid_size = int(round(num_patches ** 0.5))
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"num_patches ({num_patches}) is not a perfect square; "
                f"cannot reshape into a square attribution grid."
            )
        attr_grid = attribution.view(grid_size, grid_size)

        # Z-score + ReLU for contrast (consistent with your other visualization fns)
        mean_a = attr_grid.mean()
        std_a = attr_grid.std() + 1e-8
        attr_grid = F.relu((attr_grid - mean_a) / std_a)

        # 7. Upsample to image resolution
        B, C, H, W = image.shape
        heatmap_resized = F.interpolate(
            attr_grid.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()

        # 8. Plot original image + attribution overlay
        img_tensor = image[0].cpu().detach()
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
        img_np = img_tensor.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        overlay = ax.imshow(heatmap_resized, cmap='jet', alpha=0.60)

        plt.title(f"Integrated Gradients ({steps} steps, baseline='{baseline_mode}')\nfor Trait: '{trait_text}'",
                fontsize=11, pad=10)
        plt.axis('off')
        fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"[SUCCESS] Integrated Gradients map saved to: {save_path}")
class Bridge(nn.Module):

    def __init__(self, vision_dim: int, hidden_dim: int):
        super().__init__()

        self.projection = nn.Linear(vision_dim, hidden_dim)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.projection(image_features)
