import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

SEED_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LABEL_MAPPING = {-1: 0, 0: 1, 1: 2} # 0: Negative, 1: Neutral, 2: Positive

def load_seed_subject(mat_file_path, window_size=400, step_size=200):
    mat = sio.loadmat(mat_file_path)
    trials = [k for k in mat.keys() if not k.startswith('__')]

    trials = sorted(trials, key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    
    X_list, y_list = [], []
    
    for i, trial_name in enumerate(trials[:15]):
        raw_data = mat[trial_name] # فرمت: (62, Time_Points)
        label = LABEL_MAPPING[SEED_LABELS[i]]
        
        # پنجره‌بندی سگمنت‌ها (Windowing)
        n_samples = raw_data.shape[1]
        for start in range(0, n_samples - window_size, step_size):
            segment = raw_data[:, start:start + window_size]
            X_list.append(segment)
            y_list.append(label)
            
    X = np.array(X_list, dtype=np.float32)[:, np.newaxis, :, :] # (N, 1, 62, 400)
    y = np.array(y_list, dtype=np.int64)
    
    # Z-score Normalization روی کانال‌ها
    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True) + 1e-6
    X = (X - mean) / std
    
    return X, y
