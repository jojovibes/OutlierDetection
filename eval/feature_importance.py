import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
output_dir = "compare_feature_visuals"
os.makedirs(output_dir, exist_ok=True)

# === Feature Importance Values ===
data = {
    'Feature': ['height', 'confidence', 'area', 'ratio', 'width', 'direction', 'velocity'],
    'GMM vs CADI': [0.0247, 0.0, 0.0226, 0.0558, 0.0, 0.0, 0.0245],
    'GMM vs AVG': [0.0244, 0.0, 0.0230, 0.0565, 0.0, 0.0, 0.0248],
    'AVG vs CADI': [0.0, 0.0, 0.0760, 0.0459, 0.0996, 0.0, 0.1536]
}

df = pd.DataFrame(data)

# Melt to long format for seaborn
df_melted = df.melt(id_vars="Feature", var_name="Comparison", value_name="Importance")

# === Grouped Horizontal Bar Plot ===
plt.figure(figsize=(8, 4))
sns.barplot(data=df_melted, x="Importance", y="Feature", hue="Comparison")
plt.title("Feature Importance Across Model Disagreements")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance_barplot.png"), dpi=300)
plt.close()

# === Optional: Heatmap ===
df_heatmap = df.set_index("Feature")
plt.figure(figsize=(6, 4))
sns.heatmap(df_heatmap, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Importance'})
plt.title("Feature Importance Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance_heatmap.png"), dpi=300)
plt.close()

print("✅ Saved barplot and heatmap to:", output_dir)
