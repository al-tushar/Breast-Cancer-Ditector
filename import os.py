import os
import shutil

# Where you extracted the zip file
SOURCE ="E:/AI/BreastCancerProject/archive"

# Your project dataset folder
DEST_0 = "E:/AI/BreastCancerProject/dataset/0"
DEST_1 = "E:/AI/BreastCancerProject/dataset/1"
    
count_0 = 0
count_1 = 0
LIMIT = 3000  # copies 3000 images per class, enough to train

print("Copying images... please wait")

for patient_id in os.listdir(SOURCE):
    patient_path = os.path.join(SOURCE, patient_id)
    if not os.path.isdir(patient_path):
        continue

    for label in ["0", "1"]:
        folder = os.path.join(patient_path, label)
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.endswith(".png"):
                continue

            if label == "0" and count_0 < LIMIT:
                shutil.copy(os.path.join(folder, fname),
                            os.path.join(DEST_0, f"{patient_id}_{fname}"))
                count_0 += 1

            elif label == "1" and count_1 < LIMIT:
                shutil.copy(os.path.join(folder, fname),
                            os.path.join(DEST_1, f"{patient_id}_{fname}"))
                count_1 += 1

        if count_0 >= LIMIT and count_1 >= LIMIT:
            break

print(f"Done! Copied {count_0} benign and {count_1} malignant images")