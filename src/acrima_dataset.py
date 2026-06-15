import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset



def encontrar_directorio_dataset(base_path):
    """Busca recursivamente las carpetas Glaucoma/Normal."""
    base = Path(base_path)
    for folder in base.rglob("*"):
        if folder.is_dir() and folder.name.lower() in ["glaucoma", "normal", "g", "n"]:
            parent = folder.parent
            hijos = [d.name.lower() for d in parent.iterdir() if d.is_dir()]
            if any("glaucoma" in h or h == "g" for h in hijos) and                any("normal" in h or h == "n" for h in hijos):
                print(f"Directorio de clases: {parent}")
                return parent
    return base

class ACRIMADataset(Dataset):
    """
    Dataset ACRIMA con normalización ℓ∞ por observación.

    Normalización: x_norm = x / max(x)
    - Garantiza valores ∈ [0, 1]
    - Preserva distribución relativa de intensidades por imagen
    - Es una normalización local (por observación), no global
    """
    CLASES = ["Normal", "Glaucoma"]

    def __init__(self, root_dir, img_size=(128, 128)):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.samples, self.labels = [], []

        carpetas_clase = {}
        for d in self.root_dir.iterdir():
            if d.is_dir():
                nombre = d.name.lower()
                if "glaucoma" in nombre and "non" not in nombre:
                    carpetas_clase[1] = d

                elif (
                    "normal" in nombre
                    or "non glaucoma" in nombre
                    or nombre == "n"
                ):
                    carpetas_clase[0] = d

        if len(carpetas_clase) < 2:
            raise ValueError(f"Se necesitan 2 clases, encontradas: {list(carpetas_clase.keys())}")

        for label, carpeta in sorted(carpetas_clase.items()):
            ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            imgs = [f for f in carpeta.rglob("*") if f.suffix.lower() in ext]
            print(f"  {self.CLASES[label]}: {len(imgs)} imágenes")
            for p in imgs:
                self.samples.append((p, label))
                self.labels.append(label)

        print(f"Total: {len(self.samples)} muestras")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize(self.img_size, Image.LANCZOS)
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32)  # (H, W, 3)

        # ── NORMALIZACIÓN ℓ∞ ──────────────────────────────────────────────
        max_val = img_tensor.max()
        if max_val > 0:
            img_tensor = img_tensor / max_val   # ∈ [0, 1]
        # ──────────────────────────────────────────────────────────────────

        return img_tensor.permute(2, 0, 1), label  # (C, H, W)

