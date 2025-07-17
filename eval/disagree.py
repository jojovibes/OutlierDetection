import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import os

# INPUT_FILE = "/home/jlin1/OutlierDetection/testing/frames/output_only_logits/comparison_results_detection/combined_iou_scores.csv"  
INPUT_FILE = "/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame/detection_results_and_features.csv"
OUTPUT_DIR = "/home/jlin1/OutlierDetection/compare"
TOP_PERCENT = 0.1
TOP_N_POLARIZED = 50 

df = pd.read_csv(INPUT_FILE)
df["true_label"] = df["mask_anomaly"]

gmm_thresh = df["score_gmm"].quantile(1 - TOP_PERCENT)
avg_thresh = df["ensemble_score"].quantile(1 - TOP_PERCENT)
cadi_thresh = df["score_cadi"].quantile(1 - TOP_PERCENT)

df["pred_gmm"] = df["score_gmm"] > gmm_thresh
df["pred_ensemble"] = df["ensemble_score"] > avg_thresh
df["pred_cadi"] = df["score_cadi"] > cadi_thresh

os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_disagreement(model1, model2, label1, label2):
    pair_df = df[df[f"pred_{model1}"] != df[f"pred_{model2}"]].copy()
    pair_df[f"correct_{model1}"] = pair_df[f"pred_{model1}"] == pair_df["true_label"]
    pair_df[f"correct_{model2}"] = pair_df[f"pred_{model2}"] == pair_df["true_label"]

    pair_df["winner"] = "none"
    pair_df.loc[(pair_df[f"correct_{model1}"]) & (~pair_df[f"correct_{model2}"]), "winner"] = model1
    pair_df.loc[(pair_df[f"correct_{model2}"]) & (~pair_df[f"correct_{model1}"]), "winner"] = model2

    model1_correct = pair_df[pair_df["winner"] == model1]
    model2_correct = pair_df[pair_df["winner"] == model2]
    
    print(f"=== {label1} vs {label2} ===")
    print(f"Total disagreements: {len(pair_df)}")
    print(f"{label1} correct: {len(model1_correct)}")
    print(f"{label2} correct: {len(model2_correct)}")
    print("")

    pair_df.to_csv(os.path.join(OUTPUT_DIR, f"{label1}_vs_{label2}_disagreements.csv"), index=False)
    model1_correct.to_csv(os.path.join(OUTPUT_DIR, f"{label1}_vs_{label2}_{label1}_correct.csv"), index=False)
    model2_correct.to_csv(os.path.join(OUTPUT_DIR, f"{label1}_vs_{label2}_{label2}_correct.csv"), index=False)

     # === Analyze What Makes Them Different ===
    # compare_cols = ['velocity', 'direction', 'width', 'height', 'ratio', 'area', 'confidence']
    compare_cols = [
    "class_id",'velocity', 'direction', 'width', 'height', 'ratio', 'area', 'confidence'
]
    comparison_dir = os.path.join(OUTPUT_DIR, f"analysis_{label1}_vs_{label2}")
    os.makedirs(comparison_dir, exist_ok=True)


    for col in compare_cols:
        plt.figure()
        sns.kdeplot(data=model1_correct, x=col, label=f'{label1} Correct', fill=True)
        sns.kdeplot(data=model2_correct, x=col, label=f'{label2} Correct', fill=True)
        plt.title(f'{col} Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f'{col}_kde.png'))
        plt.close()

    # Feature importance (decision tree)
    combined = pd.concat([
        model1_correct.assign(label=0),
        model2_correct.assign(label=1)
    ])
    if not combined.empty:
        X = combined[compare_cols].dropna()
        y = combined.loc[X.index, 'label']
        if len(X) > 0 and y.nunique() > 1:
            tree = DecisionTreeClassifier(max_depth=3, random_state=0)
            tree.fit(X, y)
            importances = pd.Series(tree.feature_importances_, index=compare_cols)
            importances.sort_values(ascending=False).to_csv(os.path.join(comparison_dir, "feature_importance.csv"))
            print(f"Saved feature importances for {label1} vs {label2} to CSV")
        else:
            print(f"Not enough variation for tree-based analysis in {label1} vs {label2}")

evaluate_disagreement("gmm", "ensemble", "GMM", "AVG")
evaluate_disagreement("gmm", "cadi", "GMM", "CADI")
evaluate_disagreement("ensemble", "cadi", "AVG", "CADI")

df["diff_ensemble_gmm"] = (df["ensemble_score"] - df["score_gmm"]).abs()
df["diff_ensemble_cadi"] = (df["ensemble_score"] - df["score_cadi"]).abs()
df["diff_cadi_gmm"] = (df["score_cadi"] - df["score_gmm"]).abs()

polarization_pairs = [
    ("diff_ensemble_gmm", "ensemble_vs_GMM"),
    ("diff_ensemble_cadi", "ensemble_vs_CADI"),
    ("diff_cadi_gmm", "CADI_vs_GMM")
]

for diff_col, label in polarization_pairs:
    top_polarized = df.sort_values(by=diff_col, ascending=False).head(TOP_N_POLARIZED)
    top_polarized.to_csv(os.path.join(OUTPUT_DIR, f"top_{label}_polarized.csv"), index=False)
    print(f"Saved top {TOP_N_POLARIZED} polarized cases for {label}")