import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from eye_model import EyeModel
from eye_dataset import EyeDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

model = EyeModel()

# Prepare the dataset and dataloader
train_dataset = EyeDataset()  # Make sure YourDataset is implemented correctly
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

# Loss functions
criterion_gaze = nn.MSELoss()
criterion_blink = nn.L1Loss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.8e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# Learning rate scheduler
steps_per_epoch = len(train_loader)
scheduler = OneCycleLR(optimizer, max_lr=0.8e-3, steps_per_epoch=steps_per_epoch, epochs=60, pct_start=0.1)

# Training loop
avg_losses = []
for epoch in range(60):
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    total_loss = 0.0
    batches = 0
    for left_eye, right_eye, landmarks, gaze_est_gt, blink_prob_gt in progress_bar:
        optimizer.zero_grad()
        
        gaze_est_pred, blink_prob_pred = model(left_eye, right_eye, landmarks)
        
        loss_gaze = criterion_gaze(gaze_est_pred, gaze_est_gt) * 10  # Weighted MSE loss
        loss_blink = criterion_blink(blink_prob_pred.squeeze(), blink_prob_gt) * 15  # Weighted MAE loss
        loss = loss_gaze + loss_blink
        
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate

        total_loss += loss.item()
        batches += 1

        progress_bar.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / batches
    avg_losses.append(avg_loss)

    # 绘制每个epoch的平均loss
    plt.plot(avg_losses, label='Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.savefig("./loss.jpg")
    plt.clf()

    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f'./checkpoint_epoch_{epoch+1}.pth')