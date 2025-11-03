# ============================================================
# GLIEAM – Global–Local Interactive Emotion Analysis Model
# MULTI-CLASS VERSION (supports multiple emotion labels)
# ============================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import ViTModel, BertTokenizer, BertModel
from PIL import Image
from sklearn.metrics import classification_report

# ============================================================
# 1. DATASET LOADER (auto-detects labels dynamically)
# ============================================================

class EmotionDataset(Dataset):
    def __init__(self, root_dir, label_file, tokenizer, transform=None, limit_samples=None):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.samples = []

        if not os.path.exists(label_file):
            possible = os.path.join(root_dir, "labelResultAll.txt")
            if os.path.exists(possible):
                label_file = possible
            else:
                raise FileNotFoundError(f"❌ Label file not found: {label_file}")

        print(f"✅ Using label file: {label_file}")

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.replace('\t', ' ').replace(',', ' ').split()]
                if len(parts) >= 3:
                    idx, emotion_label, _ = parts[:3]
                    img_path = os.path.join(root_dir, f"{idx}.jpg")
                    txt_path = os.path.join(root_dir, f"{idx}.txt")

                    if os.path.exists(img_path) and os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8", errors="ignore") as t:
                            text = t.read().strip()
                        self.samples.append((img_path, text, emotion_label.lower()))

                    if limit_samples and len(self.samples) >= limit_samples:
                        break

        # Extract unique labels automatically
        unique_labels = sorted(list(set([lbl for _, _, lbl in self.samples])))
        self.label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.reverse_map = {i: lbl for lbl, i in self.label_map.items()}
        print(f"✅ Emotion classes detected: {self.label_map}")
        print(f"✅ Total valid samples loaded: {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError("❌ No valid samples found! Check your dataset paths or label format.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        enc = self.tokenizer(text, truncation=True, padding='max_length',
                             max_length=64, return_tensors='pt')
        input_ids = enc['input_ids'].squeeze()
        att_mask = enc['attention_mask'].squeeze()
        label = torch.tensor(self.label_map.get(label, 0), dtype=torch.long)
        return image, input_ids, att_mask, label


# ============================================================
# 2. GLIEAM MODEL
# ============================================================

class GLIEAM_Model(nn.Module):
    def __init__(self, num_classes):
        super(GLIEAM_Model, self).__init__()
        # Vision Transformer
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # CNN (local)
        cnn_base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(cnn_base.children())[:-2])
        self.conv_reduce = nn.Conv2d(512, 128, kernel_size=1)

        # BERT (text)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        # Fusion + classifier
        self.fc_image = nn.Linear(768 + 128, 256)
        self.fc_text = nn.Linear(768, 256)
        self.cross_fusion = nn.Linear(512, 256)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        vit_feat = self.vit(image).pooler_output
        cnn_feat = self.cnn(image)
        cnn_feat = nn.functional.adaptive_avg_pool2d(cnn_feat, (1, 1)).view(image.size(0), -1)
        cnn_feat = self.conv_reduce(cnn_feat.unsqueeze(-1).unsqueeze(-1)).view(image.size(0), -1)

        img_feat = torch.cat([vit_feat, cnn_feat], dim=1)
        img_feat = self.fc_image(img_feat)

        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = txt_out.pooler_output
        text_feat = self.fc_text(text_feat)

        fused = torch.cat([img_feat, text_feat], dim=1)
        fused = self.cross_fusion(fused)
        logits = self.classifier(fused)
        return logits


# ============================================================
# 3. TRAINING FUNCTION
# ============================================================

def train_glieam(data_root, label_file, epochs=3, batch_size=4, limit_samples=2000):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("📂 Loading dataset ...")
    dataset = EmotionDataset(data_root, label_file, tokenizer, transform, limit_samples=limit_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(dataset.label_map)
    model = GLIEAM_Model(num_classes)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("🚀 Starting training ...")
    for epoch in range(epochs):
        total_loss = 0
        for image, input_ids, att_mask, label in dataloader:
            optimizer.zero_grad()
            logits = model(image, input_ids, att_mask)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "label_map": dataset.label_map
    }, "GLIEAM_MultiEmotion.pth")
    print("✅ Training complete! Model saved as GLIEAM_MultiEmotion.pth.")


# ============================================================
# 4. EVALUATION / CLASSIFICATION FUNCTION
# ============================================================

def evaluate_glieam(model_path, data_root, label_file, limit_samples=100):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    label_map = checkpoint["label_map"]
    reverse_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = EmotionDataset(data_root, label_file, tokenizer, transform, limit_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GLIEAM_Model(num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_true, y_pred = [], []

    print("🔍 Evaluating model ...")
    with torch.no_grad():
        for image, input_ids, att_mask, label in dataloader:
            logits = model(image, input_ids, att_mask)
            pred = torch.argmax(logits, dim=1).item()
            y_true.append(label.item())
            y_pred.append(pred)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[reverse_map[i] for i in range(num_classes)]))




# ============================================================
# 4. EVALUATION FUNCTION WITH ACCURACY METRICS
# ============================================================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_glieam(model_path, data_root, label_file, limit_samples=100):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    label_map = checkpoint["label_map"]
    reverse_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = EmotionDataset(data_root, label_file, tokenizer, transform, limit_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GLIEAM_Model(num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_true, y_pred = [], []

    print("🔍 Evaluating model ...")
    with torch.no_grad():
        for image, input_ids, att_mask, label in dataloader:
            logits = model(image, input_ids, att_mask)
            pred = torch.argmax(logits, dim=1).item()
            y_true.append(label.item())
            y_pred.append(pred)

    # ============================================================
    # METRICS
    # ============================================================
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("\n📊 Evaluation Metrics:")
    print(f"✅ Accuracy : {acc*100:.2f}%")
    print(f"✅ Precision: {precision*100:.2f}%")
    print(f"✅ Recall   : {recall*100:.2f}%")
    print(f"✅ F1-score : {f1*100:.2f}%")

    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[reverse_map[i] for i in range(num_classes)]))

    # Save metrics to file
    import json
    results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    with open("GLIEAM_Eval_Metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("📁 Saved metrics → GLIEAM_Eval_Metrics.json")

# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    data_root = r"D:\ajith raj\2025 november\correction\IJCIA-D-25-00327\MVSA_Single\data"
    label_file = r"D:\ajith raj\2025 november\correction\IJCIA-D-25-00327\MVSA_Single\labelResultAll.txt"

    # Train on small subset for speed
    train_glieam(data_root, label_file, epochs=50, batch_size=4, limit_samples=2000)

    # Evaluate trained model
    evaluate_glieam("GLIEAM_MultiEmotion.pth", data_root, label_file, limit_samples=50)
