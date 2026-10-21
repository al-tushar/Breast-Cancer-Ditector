import customtkinter as ctk
from tkinter import filedialog
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# =====================================================
# MODEL
# =====================================================

class BreastCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(1)

# =====================================================
# LOAD MODEL
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = BreastCancerCNN().to(device)

model.load_state_dict(
    torch.load(
        "E:/AI/BreastCancerProject/breast_cancer_cnn.pth",
        map_location=device
    )
)

model.eval()

# =====================================================
# THEME
# =====================================================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

GOLD = "#D4AF37"
DARK_GOLD = "#B8860B"

BG = "#FFFFFF"
CARD = "#FAFAFA"
TEXT = "#222222"

# =====================================================
# APP
# =====================================================

root = ctk.CTk()

root.title("Breast Cancer Detection AI")

root.geometry("1200x750")
root.minsize(900, 650)

selected_path = {"path": None}

# =====================================================
# DARK MODE
# =====================================================

dark_enabled = {"value": False}

def toggle_theme():

    if dark_enabled["value"]:

        ctk.set_appearance_mode("light")

        dark_enabled["value"] = False

        theme_btn.configure(
            text="🌙 Dark Mode"
        )

    else:

        ctk.set_appearance_mode("dark")

        dark_enabled["value"] = True

        theme_btn.configure(
            text="☀ Light Mode"
        )

# =====================================================
# GRID CONFIG
# =====================================================

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# =====================================================
# HEADER
# =====================================================

header = ctk.CTkFrame(
    root,
    corner_radius=0,
    fg_color="transparent"
)

header.grid(
    row=0,
    column=0,
    sticky="ew",
    padx=20,
    pady=15
)

header.grid_columnconfigure(0, weight=1)

title_frame = ctk.CTkFrame(
    header,
    fg_color="transparent"
)

title_frame.grid(
    row=0,
    column=0,
    sticky="w"
)

title = ctk.CTkLabel(
    title_frame,
    text="🎗 Breast Cancer Detection AI",
    font=("Segoe UI", 30, "bold")
)

title.pack(anchor="w")

subtitle = ctk.CTkLabel(
    title_frame,
    text="Histopathology Image Classification using CNN",
    font=("Segoe UI", 14),
    text_color="gray"
)

subtitle.pack(anchor="w")

theme_btn = ctk.CTkButton(
    header,
    text="🌙 Dark Mode",
    width=140,
    height=40,
    command=toggle_theme,
    fg_color=GOLD,
    hover_color=DARK_GOLD,
    text_color="black"
)

theme_btn.grid(
    row=0,
    column=1,
    padx=10
)

# =====================================================
# MAIN CONTENT
# =====================================================

main = ctk.CTkFrame(
    root,
    fg_color="transparent"
)

main.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=20,
    pady=10
)

main.grid_columnconfigure(0, weight=3)
main.grid_columnconfigure(1, weight=2)

main.grid_rowconfigure(0, weight=1)

# =====================================================
# LEFT CARD
# =====================================================

left_card = ctk.CTkFrame(
    main,
    corner_radius=20
)

left_card.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0,10)
)

preview_title = ctk.CTkLabel(
    left_card,
    text="📁 Image Preview",
    font=("Segoe UI", 22, "bold")
)

preview_title.pack(
    pady=(20,10)
)

preview_label = ctk.CTkLabel(
    left_card,
    text="📁\n\nNo Image Selected",
    width=350,
    height=350,
    corner_radius=15,
    fg_color=("white", "#2b2b2b"),
    font=("Segoe UI", 18)
)

preview_label.pack(
    expand=True,
    pady=20
)

file_label = ctk.CTkLabel(
    left_card,
    text="",
    text_color="gray"
)

file_label.pack(
    pady=(0,20)
)
# =====================================================
# RIGHT CARD
# =====================================================

right_card = ctk.CTkFrame(
    main,
    corner_radius=20
)

right_card.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(10,0)
)

# =====================================================
# RESULT TITLE
# =====================================================

result_title = ctk.CTkLabel(
    right_card,
    text="🤖 AI Prediction",
    font=("Segoe UI", 22, "bold")
)

result_title.pack(
    pady=(20,10)
)

# =====================================================
# STATUS BADGE
# =====================================================

status_badge = ctk.CTkLabel(
    right_card,
    text="READY",
    width=140,
    height=40,
    corner_radius=20,
    fg_color="#E8F5E9",
    text_color="#2E7D32",
    font=("Segoe UI", 14, "bold")
)

status_badge.pack(
    pady=10
)

# =====================================================
# RESULT LABEL
# =====================================================

result_label = ctk.CTkLabel(
    right_card,
    text="Waiting For Analysis",
    font=("Segoe UI", 28, "bold"),
    text_color="gray"
)

result_label.pack(
    pady=20
)

# =====================================================
# CONFIDENCE
# =====================================================

confidence_title = ctk.CTkLabel(
    right_card,
    text="Confidence Score",
    font=("Segoe UI", 20, "bold")
)

confidence_title.pack(
    pady=(10,5)
)

confidence_text = ctk.CTkLabel(
    right_card,
    text="0%",
    font=("Segoe UI", 24, "bold")
)

confidence_text.pack(
    pady=(0,10)
)

# =====================================================
# PROGRESS BAR
# =====================================================

progress = ctk.CTkProgressBar(
    right_card,
    width=320,
    height=20,
    corner_radius=10,
    progress_color=GOLD
)

progress.pack(
    pady=15
)

progress.set(0)

# =====================================================
# MODEL INFO CARD
# =====================================================

info_frame = ctk.CTkFrame(
    right_card,
    corner_radius=15
)

info_frame.pack(
    fill="x",
    padx=20,
    pady=20,
)

model_info = ctk.CTkLabel(
    info_frame,
    text=(
        f"🧠 Model : Version\n"
        f"📐 Input Size : 50×50\n"
        f"💻 Device : {device}"
    ),
    justify="left",
    font=("Segoe UI", 10),
    text_color="gray"
)


model_info.pack(
    padx=15,
    pady=15,
    anchor="w"
)

# =====================================================
# IMAGE BROWSE FUNCTION
# =====================================================

def browse_image():

    path = filedialog.askopenfilename(
        title="Select Histopathology Image",
        filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.bmp")
        ]
    )

    if not path:
        return

    selected_path["path"] = path

    file_label.configure(
        text=path.split("/")[-1]
    )

    img = Image.open(path).convert("RGB")

    img.thumbnail((450,450))

    preview_image = ctk.CTkImage(
        light_image=img,
        dark_image=img,
        size=img.size
    )

    preview_label.configure(
        image=preview_image,
        text=""
    )

    preview_label.image = preview_image

    result_label.configure(
        text="Ready To Analyze",
        text_color="gray"
    )

    status_badge.configure(
        text="READY",
        fg_color="#E8F5E9",
        text_color="#2E7D32"
    )

    confidence_text.configure(
        text="0%"
    )

    progress.set(0)
    # =====================================================
# PREDICTION
# =====================================================

def perform_prediction():

    img = Image.open(
        selected_path["path"]
    ).convert("RGB")

    img = img.resize((50, 50))

    img_array = np.array(img) / 255.0

    tensor = torch.tensor(
        img_array,
        dtype=torch.float32
    )

    tensor = tensor.permute(
        2, 0, 1
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor).item()

    label = (
        "MALIGNANT"
        if prediction > 0.5
        else "BENIGN"
    )

    confidence = (
        prediction
        if prediction > 0.5
        else 1 - prediction
    )

    if label == "BENIGN":

        result_label.configure(
            text="✓ BENIGN",
            text_color="#2E7D32"
        )

        status_badge.configure(
            text="BENIGN",
            fg_color="#E8F5E9",
            text_color="#2E7D32"
        )

    else:

        result_label.configure(
            text="⚠ MALIGNANT",
            text_color="#D32F2F"
        )

        status_badge.configure(
            text="MALIGNANT",
            fg_color="#FFEBEE",
            text_color="#D32F2F"
        )

    confidence_text.configure(
        text=f"{confidence*100:.2f}%"
    )

    progress.set(confidence)


# =====================================================
# ANALYZE BUTTON
# =====================================================

def run_prediction():

    if not selected_path["path"]:

        result_label.configure(
            text="Select Image First",
            text_color="orange"
        )

        status_badge.configure(
            text="NO IMAGE",
            fg_color="#FFF3CD",
            text_color="#856404"
        )

        return

    result_label.configure(
        text="🤖 Analyzing...",
        text_color=GOLD
    )

    status_badge.configure(
        text="PROCESSING",
        fg_color="#FFF8E1",
        text_color="#FF8F00"
    )

    root.after(
        1200,
        perform_prediction
    )


# =====================================================
# TOOLBAR
# =====================================================

toolbar = ctk.CTkFrame(
    root,
    height=80,
    corner_radius=15
)

toolbar.grid(
    row=2,
    column=0,
    sticky="ew",
    padx=20,
    pady=(0, 10)
)

toolbar.grid_columnconfigure(0, weight=1)
toolbar.grid_columnconfigure(1, weight=1)

upload_btn = ctk.CTkButton(
    toolbar,
    text="📁 Browse Image",
    width=220,
    height=55,
    corner_radius=20,
    fg_color=GOLD,
    hover_color=DARK_GOLD,
    text_color="black",
    font=("Segoe UI", 15, "bold"),
    command=browse_image
)

upload_btn.grid(
    row=0,
    column=0,
    padx=20,
    pady=15
)

predict_btn = ctk.CTkButton(
    toolbar,
    text="🤖 Analyze",
    width=220,
    height=55,
    corner_radius=20,
    fg_color="#111111",
    hover_color="#333333",
    font=("Segoe UI", 15, "bold"),
    command=run_prediction
)

predict_btn.grid(
    row=0,
    column=1,
    padx=20,
    pady=15
)

# =====================================================
# HOVER EFFECTS
# =====================================================

def upload_enter(event):
    upload_btn.configure(width=235)

def upload_leave(event):
    upload_btn.configure(width=220)

upload_btn.bind("<Enter>", upload_enter)
upload_btn.bind("<Leave>", upload_leave)

def predict_enter(event):
    predict_btn.configure(width=235)

def predict_leave(event):
    predict_btn.configure(width=220)

predict_btn.bind("<Enter>", predict_enter)
predict_btn.bind("<Leave>", predict_leave)

# =====================================================
# FOOTER
# =====================================================

footer = ctk.CTkLabel(
    root,
    text="Educational Use Only • Always Consult Medical Professionals",
    text_color="gray",
    font=("Segoe UI", 11)
)

footer.grid(
    row=3,
    column=0,
    pady=(0, 10)
)

# =====================================================
# START
# =====================================================

root.mainloop()