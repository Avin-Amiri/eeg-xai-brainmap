import os
import glob
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from model import EEGNet
from data_loader import load_seed_subject
from xai_engine import get_channel_attributions
from viz_mne import plot_comparative_topomap

os.makedirs("figures", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

DATA_DIR = os.environ.get("SEED_DATA_DIR", r"C:\Mine_everything\dataset\SEED\SEED_EEG\Preprocessed_EEG")
subject_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.mat")))[:6]

if len(subject_files) == 0:
    raise FileNotFoundError(f"No .mat dataset files found in: {DATA_DIR}")

print(f"Found {len(subject_files)} subject files.")

subject_session_counter = {}
all_neg_importances = []
all_pos_importances = []

for subject_path in subject_files:
    raw_filename = os.path.splitext(os.path.basename(subject_path))[0]
    
    sub_id = raw_filename.split('_')[0].zfill(2)
    subject_session_counter[sub_id] = subject_session_counter.get(sub_id, 0) + 1
    session_num = subject_session_counter[sub_id]
    
    formatted_name = f"sub{sub_id}-se{session_num}"
    
    print(f"\n--- Processing: {formatted_name} (File: {raw_filename}) ---")
    
    X, y = load_seed_subject(subject_path, window_size=400, step_size=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=64, shuffle=False)

    model = EEGNet(n_classes=3, channels=62, samples=400).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(12):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == by).sum().item()
            total += by.size(0)
        print(f"Epoch {epoch+1}/12 - Loss: {total_loss/len(train_loader):.4f} - Acc: {correct/total*100:.2f}%")

    idx_neg = np.where(y_test == 0)[0][:10]
    idx_pos = np.where(y_test == 2)[0][:10]

    neg_tensors = torch.tensor(X_test[idx_neg])
    pos_tensors = torch.tensor(X_test[idx_pos])

    neg_imp = np.mean([get_channel_attributions(model, neg_tensors[i:i+1], target_class=0, device=device) for i in range(len(idx_neg))], axis=0)
    pos_imp = np.mean([get_channel_attributions(model, pos_tensors[i:i+1], target_class=2, device=device) for i in range(len(idx_pos))], axis=0)

    all_neg_importances.append(neg_imp)
    all_pos_importances.append(pos_imp)
    
    subj_fig_path = f"figures/{formatted_name}.png"
    plot_comparative_topomap(neg_imp, pos_imp, save_path=subj_fig_path)

mean_neg_importance = np.mean(all_neg_importances, axis=0)
mean_pos_importance = np.mean(all_pos_importances, axis=0)

print("\nGenerating Group-Averaged Brain Topography...")
plot_comparative_topomap(mean_neg_importance, mean_pos_importance, save_path="figures/brain_xai_group_average.png")
print("Done! All individual maps and group-average map saved in figures/")
