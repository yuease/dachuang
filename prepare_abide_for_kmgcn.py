import os
import numpy as np
import scipy.io as sio
import pandas as pd

# ======================
# 路径
# ======================
ABIDE_ROOT = r"C:\Users\yly\Desktop\dachuang\ABIDE\data\ABIDE_pcp\cpac\filt_noglobal"
PHENOTYPE_CSV = r"C:\Users\yly\Desktop\dachuang\ABIDE\data\ABIDE_pcp\Phenotypic_V1_0b_preprocessed1.csv"
SUBJECT_LIST = r"C:\Users\yly\Desktop\dachuang\ABIDE\data\subject_IDs.txt"
OUTPUT_FILE = r"C:\Users\yly\Desktop\dachuang\ABIDE\data\abide_kmgcn_871.npy"

# KMGCN / ABIDE 参数
N_ROI = 200
N_TIME = 78

# ======================
# 读取你项目中筛好的 subject（871）
# ======================
with open(SUBJECT_LIST, "r") as f:
    subject_ids = [line.strip() for line in f if line.strip().isdigit()]

print(f"Using {len(subject_ids)} subjects from subject_IDs.txt")

# ======================
# 读取表型信息
# ======================
pheno = pd.read_csv(PHENOTYPE_CSV)

def get_pheno(sub_id):
    row = pheno[pheno["SUB_ID"] == int(sub_id)]
    if len(row) == 0:
        return None
    label = int(row["DX_GROUP"].values[0]) - 1
    age = float(row["AGE_AT_SCAN"].values[0])
    gender = int(row["SEX"].values[0])
    return label, age, gender

# ======================
# 主循环
# ======================
final_ts, final_corr, final_pcorr = [], [], []
final_label, final_age, final_gender = [], [], []
used_subjects = []

for sub in subject_ids:
    sub_dir = os.path.join(ABIDE_ROOT, sub)
    if not os.path.isdir(sub_dir):
        continue

    ts_file = corr_file = pcorr_file = None
    for f in os.listdir(sub_dir):
        if f.endswith("_rois_cc200.1D"):
            ts_file = os.path.join(sub_dir, f)
        elif f.endswith("_cc200_correlation.mat"):
            corr_file = os.path.join(sub_dir, f)
        elif f.endswith("_cc200_partial_correlation.mat"):
            pcorr_file = os.path.join(sub_dir, f)

    if ts_file is None or corr_file is None or pcorr_file is None:
        continue

    # ROI time-series
    ts = np.loadtxt(ts_file)
    if ts.shape[0] < N_TIME:
        continue
    ts = ts[:N_TIME].T

    # corr / pcorr
    corr = list(sio.loadmat(corr_file).values())[-1]
    pcorr = list(sio.loadmat(pcorr_file).values())[-1]

    if corr.shape != (N_ROI, N_ROI) or pcorr.shape != (N_ROI, N_ROI):
        continue

    ph = get_pheno(sub)
    if ph is None:
        continue
    label, age, gender = ph

    if np.isnan(ts).any() or np.isnan(corr).any() or np.isnan(pcorr).any():
        continue

    final_ts.append(ts)
    final_corr.append(corr)
    final_pcorr.append(pcorr)
    final_label.append(label)
    final_age.append(age)
    final_gender.append(gender)
    used_subjects.append(sub)

# ======================
# 保存
# ======================
final_ts = np.stack(final_ts)
final_corr = np.stack(final_corr)
final_pcorr = np.stack(final_pcorr)

data = {
    "timeseires": final_ts,
    "corr": final_corr,
    "pcorr": final_pcorr,
    "label": np.array(final_label),
    "age": np.array(final_age),
    "gender": np.array(final_gender),
}

np.save(OUTPUT_FILE, data)

print("Final dataset:")
print("Subjects used:", len(used_subjects))
print("Time-series:", final_ts.shape)
print(f"Saved to {OUTPUT_FILE}")
