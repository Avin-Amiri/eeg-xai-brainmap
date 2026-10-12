import mne
import numpy as np
import matplotlib.pyplot as plt

SEED_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

def plot_comparative_topomap(neg_scores, pos_scores, save_path="figures/brain_xai_comparison.png"):
    ch_names = [ch.upper() for ch in SEED_CHANNELS]
    montage = mne.channels.make_standard_montage('standard_1020')
    valid_channels = [ch for ch in ch_names if ch in montage.ch_names]
    indices = [ch_names.index(ch) for ch in valid_channels]
    
    neg_filtered = np.array(neg_scores)[indices]
    pos_filtered = np.array(pos_scores)[indices]
    
    vlim = (
        min(neg_filtered.min(), pos_filtered.min()),
        max(neg_filtered.max(), pos_filtered.max())
    )
    
    info = mne.create_info(ch_names=valid_channels, sfreq=200, ch_types='eeg')
    info.set_montage(montage)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    im1, _ = mne.viz.plot_topomap(
        neg_filtered, info, axes=axes[0], show=False, 
        cmap='RdBu_r', vlim=vlim, contours=4, sphere=0.095
    )
    axes[0].set_title("Negative Emotion Relevance\n(Right-Hemisphere / Temporal Dominance)", fontsize=11, fontweight='bold')
    
    im2, _ = mne.viz.plot_topomap(
        pos_filtered, info, axes=axes[1], show=False, 
        cmap='RdBu_r', vlim=vlim, contours=4, sphere=0.095
    )
    axes[1].set_title("Positive Emotion Relevance\n(Left-Frontal Dominance)", fontsize=11, fontweight='bold')
    
    plt.colorbar(im2, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.6, label='Normalized Relevance Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Topomap saved successfully at: {save_path}")
    plt.show()
