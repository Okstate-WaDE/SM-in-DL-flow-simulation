import pandas as pd
import matplotlib.pyplot as plt
path = "/Users/gautam/Research/codesforresearch/percentsaturation/after first review/Voronoi/NSEvariation(Baseline Vs soil Moisture).xlsx"
df = pd.read_excel(path, sheet_name="SeedVsNSE graph (Voronoi) ")

x=df["Sequence Length"]
y1 = df ["RMSE_Baseline_Test"]
y2 = df ["RMSE_SM_Test"]
seq_labels = ["10","30", "90", "120", "210", "365"]
plt.figure(figsize=(7,5))
plt.plot(seq_labels, y1, label = "Baseline",  color = "#E69F00", marker ="o")
plt.plot(seq_labels, y2, label = "With Soil Moisture", color = "#2ca25f", marker = "s")
plt.xlabel (" Sequence Length (days)")
plt.ylabel ("Root Mean Square Error (RMSE) (m$^3$/sec)")


#plt.title("RMSE Vs Sequence Length (Test)")
plt.legend(loc = "upper right")
plt.grid()

plt.tight_layout()
plt.savefig("hydrograph.png", dpi=600, bbox_inches='tight')
plt.show()
