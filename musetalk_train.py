
import cv2
import os
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pickle
import glob

from .vae import VAE
from .unet import UNet
from .import musetalk_global_data

import folder_paths

image_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def preprocess_img(cv2_img_frame, image_size=256, device="cuda"):
    window = []
    if isinstance(cv2_img_frame, str):
        window_fnames = [cv2_img_frame]
        for fname in window_fnames:
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size),
                             interpolation=cv2.INTER_LANCZOS4)
            window.append(img)
    else:
        img = cv2.cvtColor(cv2_img_frame, cv2.COLOR_BGR2RGB)
        window.append(img)
    x = np.asarray(window) / 255.
    x = np.transpose(x, (3, 0, 1, 2))
    x = torch.squeeze(torch.FloatTensor(x))
    x = image_transform(x)
    # x = x.unsqueeze(0)  # [1, 3, 256, 256] torch tensor
    x = x.to(device)
    return x


class FaceDataset(Dataset):
    def __init__(self, face_latents, audio_features, cv2_frames):

        super(FaceDataset, self).__init__()

        self.face_latents = face_latents
        self.audio_features = audio_features
        self.cv2_frames =  cv2_frames

        print('FaceDataset', len(self.face_latents))

    def __getitem__(self, item):

        frame_tensor = preprocess_img(self.cv2_frames[item])
        latent = self.face_latents[item].squeeze(0)
        audio_feature = self.audio_features[item]
        audio_feature = torch.tensor(audio_feature).cuda()

        # print(f"frame_tensor: {frame_tensor.shape}, latent: {latent.shape}, audio_feature: {audio_feature.shape}")
        return frame_tensor, latent, audio_feature

    def __len__(self):
        return len(self.face_latents)

# only for debug
class FaceDataset2(Dataset):
    def __init__(self, dataset_root):
        super(FaceDataset2, self).__init__()
        self.dataset_root = dataset_root
        self.frame_root = os.path.join(self.dataset_root, "frame")
        with open(os.path.join(self.dataset_root, "face_latent.pkl"), 'rb') as f:
            self.face_latents = pickle.load(f)
        with open(os.path.join(self.dataset_root, "whisper_chunks.pkl"), 'rb') as f:
            self.audio_features = pickle.load(f)
        self.frames_im_path_list = list(sorted(glob.glob(os.path.join(self.frame_root, "*.png"))))

    def __getitem__(self, item):
        frame = cv2.imread(self.frames_im_path_list[item])
        frame_tensor = preprocess_img(frame)
        latent = self.face_latents[item].squeeze(0)
        audio_feature = self.audio_features[item]
        audio_feature = torch.tensor(audio_feature).cuda()

        # print(f"frame_tensor: {frame_tensor.shape}, latent: {latent.shape}, audio_feature: {audio_feature.shape}")
        return frame_tensor, latent, audio_feature

    def __len__(self):
        return len(self.frames_im_path_list)


class MuseTalkTrain:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "whisper_features" : ("WHISPERFEAT",),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )

    FUNCTION = "train"
    CATEGORY = "MuseTalkUtils"

    # TODO, images
    def train(self, images, whisper_features, batch_size):

        with torch.inference_mode(False):

            model_path_base = os.path.join(folder_paths.models_dir,'musetalk')
            model_config_path = os.path.join(model_path_base, "musetalk", "musetalk.json")
            model_bin_path = os.path.join(model_path_base, "musetalk", "pytorch_model.bin")# TODO, name
            vae_path = os.path.join(model_path_base, "sd-vae-ft-mse")

            # model_config_path = "F:/MuseTalk/talk/models/musetalk/musetalk.json"
            # model_bin_path = "F:/MuseTalk/talk/models/musetalk/pytorch_model.bin"
            # vae_path = "F:/MuseTalk/talk/models/sd-vae-ft-mse/"

            unet = UNet(unet_config = model_config_path, model_path = model_bin_path)
            vae = VAE(model_path = vae_path)

            # global unet
            global resized_cv2_frame_list
            global faces_latent_list

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            vae.vae.eval()
            unet.model.train()

            timesteps = torch.tensor([0], device=device)
            lr = 1e-4
            # lr = 5e-5
            criterion = nn.HuberLoss()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, unet.model.parameters()), lr=lr)
            # optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, unet.model.parameters()), lr=lr)

            # 
            save_ckpt_dir = os.path.join(model_path_base, "musetalk")

            print("len", len(musetalk_global_data.faces_latent_list), len(whisper_features), len(musetalk_global_data.resized_cv2_frame_list))

            face_dataset = FaceDataset(musetalk_global_data.faces_latent_list, whisper_features, musetalk_global_data.resized_cv2_frame_list)

            # face_dataset = FaceDataset2("F:/MuseTalk/talk/data/train_dataset/v2")

            face_dataloader = DataLoader(face_dataset, batch_size = batch_size, shuffle=True, num_workers=0)

            # TODO param
            for epoch in range(0, 100):
                pbar = tqdm.tqdm(enumerate(face_dataloader), total=len(face_dataloader))
                loss_log = []
                for i, (face_tensor, latent_tensor, audio_feat) in pbar:

                    audio_feat = audio_feat.to(torch.float32)

                    pred_latents = unet.model(latent_tensor, timesteps, encoder_hidden_states=audio_feat).sample

                    # print(f"pred_latents: {pred_latents.requires_grad}")
                    recon = vae.just_decode_latents(pred_latents)

                    gt_latent = vae.encode_latents(face_tensor)
                    loss = 0.2 * criterion(pred_latents, gt_latent) + 0.8 * criterion(recon, face_tensor)
                    loss.backward()
                    loss_log.append(loss.item())
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.set_description("(Epoch {}) TRAIN LOSS:{:.8f}".format((epoch + 1), np.mean(loss_log)))

                torch.save(unet.model.state_dict(), os.path.join(save_ckpt_dir, "epoch_{}.pth".format(epoch)))

        return (images,)

if __name__ == "__main__":

    train = MuseTalkTrain()
    train.train(None, [], 4)


# print("dgdg")

# print("hehehh ")
# train = MuseTalkTrain()
# train.train(None, [], 4)
