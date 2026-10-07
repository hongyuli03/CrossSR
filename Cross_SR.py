import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
import cv2
import shutil
import random
from tqdm import tqdm


from timm.layers import PatchEmbed
from timm.models.vision_transformer import Block


VAL_LR_DIR = "/path/to/reference_lr_dataset"
DATASET_CONFIG = {
    "/path/to/candidate_dataset_1": "Dataset1",
    "/path/to/candidate_dataset_2": "Dataset2",
    "/path/to/candidate_dataset_3": "Dataset3",
    "/path/to/candidate_dataset_4": "Dataset4",
    "/path/to/candidate_dataset_5": "Dataset5",
    "/path/to/candidate_dataset_6": "Dataset6",
}
BASE_DIR = "/path/to/project_root"
OUTPUT_DIR = os.path.join(BASE_DIR, "selected_training_set")
LIST_FILE_PATH = os.path.join(BASE_DIR, "selection_results.txt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "mae_model.pth")




class TimmTinyMAE(nn.Module):
    def __init__(self, img_size=140, patch_size=14, in_chans=3, embed_dim=128, enc_depth=4, dec_depth=2, num_heads=4):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(enc_depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)


        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(dec_depth)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)


        self.pred_head = nn.Linear(embed_dim, patch_size ** 2 * in_chans)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, imgs, mask_ratio=0.75):

        x = self.patch_embed(imgs)
        x = x + self.pos_embed


        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)


        for blk in self.blocks:
            x_masked = blk(x_masked)
        x_masked = self.norm(x_masked)


        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x_masked.shape[1], 1)
        x_ = torch.cat([x_masked, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x_ = x_ + self.decoder_pos_embed


        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)


        pred = self.pred_head(x_)


        p = self.patch_embed.patch_size[0]
        h = w = imgs.shape[2] // p
        target = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        target = torch.einsum('nchpwq->nhwpqc', target)
        target = target.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss, pred, mask



def smart_process_lq(img_pil, crop_size=140):
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h > 0 or pad_w > 0:
        img_np = cv2.copyMakeBorder(img_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_new, w_new = img_np.shape[:2]
    patches = []
    stride = 70
    for y in range(0, h_new - crop_size + 1, stride):
        for x in range(0, w_new - crop_size + 1, stride):
            patch = img_np[y:y + crop_size, x:x + crop_size]
            patches.append(Image.fromarray(patch))
    if not patches and h_new >= crop_size and w_new >= crop_size:
        patches.append(Image.fromarray(img_np[:crop_size, :crop_size]))
    return patches


def process_train_image_downsampled(img_pil, crop_size=140):
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    target_h, target_w = h // 8, w // 8
    if target_h < crop_size or target_w < crop_size:
        scale = max(crop_size / h, crop_size / w)
        if scale < 1.0:
            img_resized = cv2.resize(img_np, (max(target_w, crop_size), max(target_h, crop_size)),
                                     interpolation=cv2.INTER_CUBIC)
        else:
            img_resized = img_np
    else:
        img_resized = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    h_new, w_new = img_resized.shape[:2]
    pad_h = (crop_size - (h_new % crop_size)) % crop_size
    pad_w = (crop_size - (w_new % crop_size)) % crop_size
    if pad_h > 0 or pad_w > 0:
        img_resized = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    h_final, w_final = img_resized.shape[:2]
    patches = []
    for y in range(0, h_final, crop_size):
        for x in range(0, w_final, crop_size):
            patch = img_resized[y:y + crop_size, x:x + crop_size]
            patches.append(Image.fromarray(patch))
    return patches


def get_image_paths_with_dataset_info(config_dict):
    all_files = []
    for dir_path, prefix in config_dict.items():
        if not os.path.exists(dir_path): continue
        types = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        files_grabbed = []
        for t in types:
            files_grabbed.extend(glob.glob(os.path.join(dir_path, t)))
        for fpath in files_grabbed:
            all_files.append({"path": fpath, "prefix": prefix, "basename": os.path.basename(fpath)})
    return all_files



if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_ratio = 0.75

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(f"TAMP-MAE | device={device} | mask_ratio={mask_ratio}")




    val_files = sorted(glob.glob(os.path.join(VAL_LR_DIR, "*.*")))
    val_files = [f for f in val_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]


    random.seed(42)
    random.shuffle(val_files)
    num_val = max(5, len(val_files) // 10)
    val_set_files = val_files[:num_val]
    train_set_files = val_files[num_val:]


    def extract_and_augment_tensors(file_list, augment=False):
        tensors = []
        for p in file_list:
            crops = smart_process_lq(Image.open(p).convert('RGB'), crop_size=140)
            for crop in crops:
                if augment:
                    if random.random() > 0.5: crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                    if random.random() > 0.5: crop = crop.transpose(Image.FLIP_TOP_BOTTOM)
                tensors.append(transform(crop))
        return torch.stack(tensors) if tensors else None



    train_tensors = extract_and_augment_tensors(train_set_files, augment=True)
    val_tensors = extract_and_augment_tensors(val_set_files, augment=False)

    if train_tensors is None or val_tensors is None:
        raise ValueError("Failed to load reference images from VAL_LR_DIR.")

    train_tensors = train_tensors.to(device)
    val_tensors = val_tensors.to(device)
    print(f"Train patches: {train_tensors.shape[0]} | Val patches: {val_tensors.shape[0]}")


    model = TimmTinyMAE(
        img_size=140,
        patch_size=14,
        embed_dim=128,
        enc_depth=4,
        dec_depth=2,
        num_heads=4,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    batch_size = 32
    max_epochs = 100
    patience = 5

    best_val_loss = float('inf')
    patience_counter = 0

    print("Training MAE...")
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        indices = torch.randperm(train_tensors.shape[0])

        for i in range(0, train_tensors.shape[0], batch_size):
            batch = train_tensors[indices[i:i + batch_size]]
            optimizer.zero_grad()
            loss, _, _ = model(batch, mask_ratio=mask_ratio)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)

        avg_train_loss = train_loss / train_tensors.shape[0]


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, val_tensors.shape[0], batch_size):
                batch = val_tensors[i:i + batch_size]
                v_loss, _, _ = model(batch, mask_ratio=mask_ratio)
                val_loss += v_loss.item() * batch.size(0)

        avg_val_loss = val_loss / val_tensors.shape[0]
        print(f"Epoch {epoch + 1:03d}/{max_epochs} | train_loss={avg_train_loss:.5f} | val_loss={avg_val_loss:.5f}")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}; best model saved.")
                break




    print("Scoring candidate images...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    train_info_list = get_image_paths_with_dataset_info(DATASET_CONFIG)
    train_results_db = []

    with torch.no_grad():
        for item in tqdm(train_info_list, desc="Scoring Candidates"):
            fpath = item["path"]
            try:
                img = Image.open(fpath).convert('RGB')
                patches = process_train_image_downsampled(img, crop_size=140)
                if not patches: continue

                patch_tensors = torch.stack([transform(p) for p in patches]).to(device)

                loss_list = []
                for i in range(0, patch_tensors.shape[0], batch_size):
                    batch = patch_tensors[i:i + batch_size]
                    loss, _, _ = model(batch, mask_ratio=mask_ratio)
                    loss_list.append(loss.item())


                best_patch_score = max([1.0 / (mse + 1e-6) for mse in loss_list])
                item["tamp_score"] = best_patch_score
                train_results_db.append(item)

            except Exception as e:
                continue




    top_k = 1000
    print(f"Selecting top {top_k} images...")
    train_results_db.sort(key=lambda x: x["tamp_score"], reverse=True)
    selected_items = train_results_db[:top_k]

    with open(LIST_FILE_PATH, "w") as f:
        f.write("Rank\tTAMP_Score\tNew_Name\tOriginal_Path\n")
        for rank, item in enumerate(tqdm(selected_items, desc="Copying Winners")):
            rank_idx = rank + 1
            new_name = f"{item['prefix']}_Rank{rank_idx}_{item['basename']}"
            f.write(f"{rank_idx}\t{item['tamp_score']:.2f}\t{new_name}\t{item['path']}\n")

            try:
                shutil.copy2(item["path"], os.path.join(OUTPUT_DIR, new_name))
            except Exception as e:
                pass

    print("TAMP-MAE selection complete.")

import os
import cv2
import numpy as np
import pywt
import glob
from tqdm import tqdm


VAL_LR_DIR = "/path/to/reference_lr_dataset"


INPUT_HR_DIR = os.path.join(BASE_DIR, "selected_training_set")


OUTPUT_DIR = os.path.join(BASE_DIR, "aligned_training_set")





def list_image_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.*")))
    files = [p for p in files if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    return files



def compute_global_ab_ll_proto(ref_paths, wavelet='haar', use_resize=False, resize_to=128):
    ab_ll_values = {1: [], 2: []}

    for ref_path in tqdm(ref_paths, desc="Building Global Proto"):
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            continue

        if use_resize:
            ref_img = cv2.resize(ref_img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

        ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        for c in [1, 2]:
            channel = ref_lab[:, :, c]
            LL, _ = pywt.dwt2(channel, wavelet)
            ab_ll_values[c].append(LL.reshape(-1))

    proto_stats = {}

    for c in [1, 2]:
        if len(ab_ll_values[c]) == 0:
            raise ValueError(f"No valid reference data found for channel {c}.")

        all_vals = np.concatenate(ab_ll_values[c], axis=0)
        mu = float(np.mean(all_vals))
        sigma = float(np.std(all_vals) + 1e-5)
        proto_stats[c] = (mu, sigma)

    return proto_stats



def wavelet_ab_alignment_with_global_proto(src_img, proto_stats, wavelet='haar',
                                           min_scale=0.4, max_scale=2.5, min_std=1e-2):
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)


    aligned_lab = src_lab.copy()


    for c in [1, 2]:
        src_channel = src_lab[:, :, c]


        LL_src, (LH_src, HL_src, HH_src) = pywt.dwt2(src_channel, wavelet)

        src_mean = np.mean(LL_src)
        src_std = np.std(LL_src)

        ref_mean, ref_std = proto_stats[c]


        safe_src_std = max(src_std, min_std)
        scale = ref_std / safe_src_std
        scale = np.clip(scale, min_scale, max_scale)


        LL_aligned = (LL_src - src_mean) * scale + ref_mean


        recon = pywt.idwt2((LL_aligned, (LH_src, HL_src, HH_src)), wavelet)

        h, w = src_lab.shape[:2]
        aligned_lab[:, :, c] = recon[:h, :w]

    aligned_lab = np.clip(aligned_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(aligned_lab, cv2.COLOR_LAB2BGR)



if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lq_paths = list_image_files(VAL_LR_DIR)
    hr_paths = list_image_files(INPUT_HR_DIR)

    print(f"Alignment | reference={len(lq_paths)} | input={len(hr_paths)}")

    if len(lq_paths) == 0:
        raise ValueError("No reference images found in VAL_LR_DIR.")

    if len(hr_paths) == 0:
        raise ValueError("No input images found in INPUT_HR_DIR.")

    wavelet = "haar"
    use_resize_for_ref = False
    ref_resize_to = 128

    print("Building global A/B prototype...")
    proto_stats = compute_global_ab_ll_proto(
        lq_paths,
        wavelet=wavelet,
        use_resize=use_resize_for_ref,
        resize_to=ref_resize_to
    )

    print(f"Prototype | A(mean={proto_stats[1][0]:.4f}, std={proto_stats[1][1]:.4f}) | B(mean={proto_stats[2][0]:.4f}, std={proto_stats[2][1]:.4f})")

    min_scale = 0.4
    max_scale = 2.5
    min_std = 1e-2

    print("Aligning selected images...")

    for hr_path in tqdm(hr_paths, desc="Global Proto Aligning"):
        src_img = cv2.imread(hr_path)
        if src_img is None:
            continue

        aligned_img = wavelet_ab_alignment_with_global_proto(
            src_img,
            proto_stats,
            wavelet=wavelet,
            min_scale=min_scale,
            max_scale=max_scale,
            min_std=min_std
        )

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(hr_path))
        cv2.imwrite(out_path, aligned_img)

    print("Alignment complete.")
