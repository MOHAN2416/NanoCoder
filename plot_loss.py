import json, matplotlib.pyplot as plt

with open("checkpoints/training_log.json") as f:
    log = json.load(f)

steps = [x["step"] for x in log]
plt.figure(figsize=(8, 4))
plt.plot(steps, [x["train_loss"] for x in log], label="Train Loss")
plt.plot(steps, [x["val_loss"] for x in log], label="Validation Loss")
plt.xlabel("Training Steps")
plt.ylabel("Cross-Entropy Loss")
plt.title("NanoCoder (12.29M) Training Dynamics")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")