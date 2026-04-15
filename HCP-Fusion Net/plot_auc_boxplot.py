import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    clin = pd.read_csv("checkpoints/cvCentral_r50_clinicalOnly_cv.csv")
    img = pd.read_csv("checkpoints/cvCentral_Bonly_r50_patientAgg.csv")
    cdr = pd.read_csv("checkpoints/cvCentral_Bonly_r50_clinPlusDeepRad_cv.csv")

    # clinical-only 用 mean+max 作为代表（可按需改成 mean/max）
    clin = clin[clin["agg"] == "mean+max"][["fold", "auc_test"]].copy()
    clin["method"] = "Clinical-only"

    # image-only 用 patient-level mean 聚合作为代表
    img = img[["fold", "auc_patient_mean"]].rename(columns={"auc_patient_mean": "auc_test"})
    img["method"] = "Image-only (B, patient-mean)"

    cdr = cdr[["fold", "auc_test"]].copy()
    cdr["method"] = "Clinical + DeepRadScore"

    df = pd.concat([clin, img, cdr], ignore_index=True)

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "auc_boxplot_three_routes.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="method", y="auc_test")
    sns.stripplot(data=df, x="method", y="auc_test", color="black", alpha=0.7, jitter=0.1)
    plt.ylabel("Test AUC (5-fold)")
    plt.xlabel("")
    plt.ylim(0.45, 0.80)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"saved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

clin = pd.read_csv("checkpoints/cvCentral_r50_clinicalOnly_cv.csv")
img = pd.read_csv("checkpoints/cvCentral_Bonly_r50_patientAgg.csv")
cdr = pd.read_csv("checkpoints/cvCentral_Bonly_r50_clinPlusDeepRad_cv.csv")

# clinical-only 用 mean+max 作为代表（你也可改成 mean/max）
clin = clin[clin["agg"] == "mean+max"][["fold", "auc_test"]].copy()
clin["method"] = "Clinical-only"

img = img[["fold", "auc_patient_mean"]].rename(columns={"auc_patient_mean": "auc_test"})
img["method"] = "Image-only (B, patient-mean)"

cdr = cdr[["fold", "auc_test"]].copy()
cdr["method"] = "Clinical + DeepRadScore"

df = pd.concat([clin, img, cdr], ignore_index=True)

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="method", y="auc_test")
sns.stripplot(data=df, x="method", y="auc_test", color="black", alpha=0.7, jitter=0.1)
plt.ylabel("Test AUC (5-fold)")
plt.xlabel("")
plt.ylim(0.45, 0.80)
plt.tight_layout()
plt.savefig("figures/auc_boxplot_three_routes.png", dpi=300)
print("saved: figures/auc_boxplot_three_routes.png")