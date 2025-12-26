import torch
import os
def save_checkpoint(model, optimizer, epoch, val_loss,title):
    torch.save({
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "val_loss":val_loss
    },f'Object_detection.{title}.pth')
    print(f"Model {title} saved successfully.")

def load_checkpoint(title, model, optimizer=None):
    if not os.path.isfile(f"Object_detection.{title}.pth"):
        return model,optimizer,-1,float('inf')
    checkpoint = torch.load(f"Object_detection.{title}.pth",map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is None:
        return model
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    return model,optimizer,epoch,val_loss
def get_best_val_loss():
    checkpoint = torch.load(f"Object_detection.best_checkpoint.pth", map_location="cuda")
    return checkpoint["val_loss"]