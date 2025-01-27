import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from src.data_loader import load_dataset

X, y = load_dataset()

# Verify data integrity
print(f"Data shape: {X.shape}")  # Should be (1000, 130, 13)
print(f"Sample MFCC range: {X[0].min():.2f} to {X[0].max():.2f}")

# Create subplot with proper labels
fig, ax = plt.subplots()
mfcc_heatmap = ax.imshow(X[0].T, 
                       origin='lower', 
                       aspect='auto',
                       cmap='viridis')

# Set axis labels
ax.set_xlabel("Time Frames (0-129)")
ax.set_ylabel("MFCC Coefficients (0-12)")
ax.set_yticks(range(13))
ax.set_xticks(range(0, 130, 20))  # Show every 20th frame

# Add colorbar with label
cbar = fig.colorbar(mfcc_heatmap)
cbar.set_label("MFCC Magnitude (dB)")

plt.title(f"Blues Sample MFCCs (Genre ID: {y[0]})")

# Save and verify
save_path = Path('/app/data/mfcc_sample.png')
plt.savefig(save_path, bbox_inches='tight', dpi=100)
plt.close()
print(f"Saved corrected plot to {save_path}")