import re
import matplotlib.pyplot as plt

# Path to your log file
log_file = "without_meanflow/log.txt"

# Prepare containers
steps, loss, mse_loss, lr = [], [], [], []

# Regex pattern to extract values
pattern = re.compile(
    r"Global Step:\s*([\d\.]+).*?Loss:\s*([\d\.]+).*?MSE_Loss:\s*([\d\.]+).*?Learning Rate:\s*([\d\.]+)"
)

# Read and parse the log file
with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            steps.append(float(match.group(1)))
            loss.append(float(match.group(2)))
            mse_loss.append(float(match.group(3)))
            lr.append(float(match.group(4)))

# Visualization
plt.figure(figsize=(10,6))

# Plot Loss curves
plt.plot(steps, loss, marker='o', label="Loss")
plt.plot(steps, mse_loss, marker='s', label="MSE_Loss")

# Learning rate on secondary y-axis
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(steps, lr, color='green', linestyle='--', label="Learning Rate")

# Labels and title
ax.set_xlabel("Global Step")
ax.set_ylabel("Loss Value")
ax2.set_ylabel("Learning Rate")
plt.title("Training Metrics Curve")

# Legends
lns1, labs1 = ax.get_legend_handles_labels()
lns2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lns1 + lns2, labs1 + labs2, loc="best")

plt.grid(True)
plt.tight_layout()
plt.show()
