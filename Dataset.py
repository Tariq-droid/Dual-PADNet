import os
import torch
from PIL import Image
from torch.utils.data import Dataset



class DualBranchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, which contains the subfolders:
                               real/, real_face/, spoof/, spoof_face/
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get the list of image file names for real and spoof
        self.samples = []
        for label_name in ['real', 'spoof']:
            context_dir = os.path.join(root_dir, label_name)
            face_dir = os.path.join(root_dir, label_name + '_face')
            # List all files in context_dir
            context_images = os.listdir(context_dir)
            # Assume images in face_dir have same names
            for img_name in context_images:
                context_img_path = os.path.join(context_dir, img_name)
                face_img_path = os.path.join(face_dir, img_name)
                if os.path.exists(face_img_path):
                    label = 1 if label_name == 'real' else 0
                    self.samples.append((context_img_path, face_img_path, label))
                else:
                    print(f"Warning: Face image {face_img_path} not found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context_img_path, face_img_path, label = self.samples[idx]
        context_image = Image.open(context_img_path).convert('RGB')
        face_image = Image.open(face_img_path).convert('RGB')
        if self.transform:
            context_image = self.transform(context_image)
            face_image = self.transform(face_image)
        label = torch.tensor(label, dtype=torch.float32)
        return face_image, context_image, label