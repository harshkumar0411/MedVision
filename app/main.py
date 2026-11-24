import os
import io
import csv
import tempfile
from datetime import datetime
import cv2
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
from fpdf import FPDF
from torchvision import models, transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# ======================================
# PAGE CONFIG & SPACIOUS MEDICAL STYLING
# ======================================
st.set_page_config(
    page_title="MedVision AI — Diagnostic Workstation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    .header-box {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.12), rgba(14, 116, 144, 0.06));
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-bottom: 1.5rem;
    }
    .header-box h1 {
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
    }
    .header-box p {
        margin: 0.35rem 0 0 0;
        opacity: 0.82;
        font-size: 0.95rem;
    }
    .result-card {
        padding: 1.25rem 1.4rem;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================
# SPECIALTY REGISTRY & CLINICAL NOTES
# ======================================
SPECIALTIES = {
    "🧠 Brain Tumor (MRI)": {
        "file": "models/brain_tumor.pth",
        "modality": "Brain MRI Scan",
        "description": "Evaluates cranial MRI slices for Glioma, Meningioma, Pituitary adenoma, or Normal parenchyma.",
    },
    "🦴 Bone Break (X-Ray)": {
        "file": "models/bone_break.pth",
        "modality": "Musculoskeletal X-Ray",
        "description": "Detects and classifies orthopedic fracture patterns across radiographic bone views.",
    },
    "🫁 Lung Cancer (CT / Scan)": {
        "file": "models/lung_cancer.pth",
        "modality": "Pulmonary Scan",
        "description": "Analyzes pulmonary scans for malignant subtypes versus benign/normal lung tissue.",
    },
    "🫘 Renal Malignancy (CT)": {
        "file": "models/renal_malignancy.pth",
        "modality": "Renal CT Scan",
        "description": "Differentiates normal kidney anatomy, cysts, calculi (stones), and renal cell tumors.",
    },
    "🔬 Skin Lesions (Dermoscopy)": {
        "file": "models/skin_lesions.pth",
        "modality": "Dermoscopic Image",
        "description": "Screens pigmented and non-pigmented cutaneous lesions across dermatological categories.",
    },
}

CLASS_NOTES = {
    "glioma": "Neoplasm arising from glial support cells in the brain or spinal cord.",
    "meningioma": "Typically extra-axial tumor originating from the meninges surrounding the brain.",
    "notumor": "No abnormal intracranial mass or focal neoplastic lesion detected in this slice.",
    "pituitary": "Sellar/parasellar lesion originating from the pituitary gland at the skull base.",
}

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


# ======================================
# MODEL LOADER (SUPPORTS B0, B2 & CONVNEXT)
# ======================================
@st.cache_resource
def load_specialty_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    classes = checkpoint["class_names"]
    img_size = checkpoint.get("image_size", 224)
    val_acc = checkpoint.get("val_accuracy", None)
    arch = checkpoint.get("arch", "efficientnet_b0")

    if arch == "efficientnet_b2":
        model = models.efficientnet_b2()
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, len(classes)),
        )
        target_layers = [model.features[-1]]
    elif arch == "convnext_tiny":
        model = models.convnext_tiny()
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, len(classes)),
        )
        target_layers = [model.features[-1]]
    elif arch == "convnext_small":
        model = models.convnext_small()
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, len(classes)),
        )
        target_layers = [model.features[-1]]
    else:
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
        target_layers = [model.features[-1]]

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, device, classes, img_size, val_acc, arch, target_layers


# ======================================
# INFERENCE & IMAGE UTILITIES
# ======================================
@torch.no_grad()
def run_inference(model, input_tensor):
    p1 = torch.softmax(model(input_tensor), dim=1)
    p2 = torch.softmax(model(torch.flip(input_tensor, dims=[3])), dim=1)
    p3 = torch.softmax(model(torch.flip(input_tensor, dims=[2])), dim=1)
    p4 = torch.softmax(model(torch.rot90(input_tensor, k=1, dims=[2, 3])), dim=1)

    single_probs = p1[0].cpu().numpy()
    tta_probs = ((p1 + p2 + p3 + p4) / 4.0)[0].cpu().numpy()
    return single_probs, tta_probs


def enhance_scan(pil_img, brightness=1.0, contrast=1.0, sharpness=1.0):
    img = ImageEnhance.Brightness(pil_img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


def draw_attention_contours(rgb_uint8, grayscale_cam, threshold=0.65):
    overlay = rgb_uint8.copy()
    mask = (grayscale_cam >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 50, 50), 2)
    return overlay, len(contours)


def create_demo_scan(size=260):
    img = Image.new("RGB", (size, size), color=(18, 20, 26))
    draw = ImageDraw.Draw(img)
    draw.ellipse([28, 22, size - 28, size - 22], fill=(75, 80, 90), outline=(140, 145, 155), width=4)
    draw.ellipse([65, 55, size - 65, size - 55], fill=(110, 115, 125))
    draw.ellipse([int(size * 0.56), int(size * 0.34), int(size * 0.74), int(size * 0.52)], fill=(215, 220, 230))
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


# ======================================
# UNICODE-SAFE PDF & CSV EXPORTERS
# ======================================
def _pdf_safe(text):
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "ignore").decode("latin-1").strip()


def generate_pdf_report(
    orig_img,
    cam_img,
    specialty,
    scan_name,
    pred_class,
    confidence,
    triage_status,
    classes,
    probs,
    arch,
    use_tta,
):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 11, _pdf_safe("MedVision AI - Diagnostic Summary"), ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    clean_specialty = _pdf_safe(specialty)
    clean_scan = _pdf_safe(scan_name)
    mode_str = "4-View TTA" if use_tta else "Single-View"
    pdf.cell(
        0,
        7,
        _pdf_safe(f"Specialty: {clean_specialty} | Scan: {clean_scan} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
        ln=True,
        align="C",
    )
    pdf.cell(0, 6, _pdf_safe(f"Backbone: {arch} | Inference Mode: {mode_str}"), ln=True, align="C")
    pdf.ln(5)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
        orig_img.save(f1.name)
        Image.fromarray(cam_img).save(f2.name)
        orig_path, cam_path = f1.name, f2.name

    pdf.image(orig_path, x=25, y=42, w=70)
    pdf.image(cam_path, x=115, y=42, w=70)
    os.remove(orig_path)
    os.remove(cam_path)

    pdf.set_y(118)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, _pdf_safe(f"Primary Finding: {pred_class.upper()} ({confidence:.2f}% Confidence)"), ln=True)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, _pdf_safe(f"Assessment: {triage_status}"), ln=True)

    note = CLASS_NOTES.get(pred_class.lower(), "")
    if note:
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _pdf_safe(f"Reference Note: {note}"))
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Class Probability Breakdown:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for cls, prob in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
        pdf.cell(0, 7, _pdf_safe(f"  - {cls}: {prob * 100:.2f}%"), ln=True)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        5,
        _pdf_safe(
            "Disclaimer: Generated by an AI research prototype using Grad-CAM++ explainability. "
            "For educational and research evaluation only; not a substitute for clinical diagnosis."
        ),
    )
    return bytes(pdf.output())


def history_to_csv(records):
    if not records:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")


# ======================================
# CLEAN SIDEBAR (PRIMARY NAVIGATION + COLLAPSED EXTRAS)
# ======================================
available_specialties = {k: v for k, v in SPECIALTIES.items() if os.path.exists(v["file"])}
if not available_specialties:
    st.error("No trained `.pth` model checkpoints found in `models/`. Run `python training/train.py` first.")
    st.stop()

with st.sidebar:
    st.header("🩺 Specialty Module")
    selected_specialty = st.selectbox(
        "Select Diagnostic Domain",
        list(available_specialties.keys()),
        label_visibility="collapsed",
    )
    spec_info = available_specialties[selected_specialty]
    model, device, CLASSES, IMG_SIZE, val_acc, arch, target_layers = load_specialty_model(spec_info["file"])
    st.caption(spec_info["description"])

    st.divider()
    use_tta = st.toggle(
        "4-View TTA Accuracy Boost",
        value=True,
        help="Averages predictions across 4 geometric views for higher stability.",
    )

    with st.expander("🖼️ Image & Heatmap Adjustments", expanded=False):
        if st.button("Reset Adjustments", use_container_width=True):
            st.session_state["brightness_slider"] = 1.0
            st.session_state["contrast_slider"] = 1.0
            st.session_state["sharpness_slider"] = 1.0
        brightness = st.slider("Brightness", 0.5, 1.8, 1.0, 0.05, key="brightness_slider")
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05, key="contrast_slider")
        sharpness = st.slider("Sharpness", 0.5, 2.5, 1.0, 0.1, key="sharpness_slider")
        cam_alpha = st.slider("Heatmap Opacity", 0.15, 0.85, 0.45, 0.05)
        contour_thresh = st.slider("Hotspot Contour Threshold", 0.30, 0.90, 0.65, 0.05)

    with st.expander("📌 Model & Hardware Details", expanded=False):
        st.write(f"**Modality:** {spec_info['modality']}")
        st.write(f"**Backbone:** `{arch}` ({IMG_SIZE}×{IMG_SIZE})")
        if val_acc is not None:
            st.write(f"**Benchmark Accuracy:** `{val_acc * 100:.2f}%`")
        st.write(f"**Classes:** {', '.join(CLASSES)}")
        st.write(f"**Device:** `{device.type.upper()}`")

    st.divider()
    st.caption("⚠️ Educational & research decision-support tool only. Not a clinical diagnosis.")

# ======================================
# SPACIOUS HEADER & UPLOAD BAR
# ======================================
st.markdown(
    f"""
    <div class="header-box">
        <h1>🏥 MedVision AI — {selected_specialty}</h1>
        <p>Upload a {spec_info['modality'].lower()} for automated classification, Grad-CAM++ localization, and PDF reporting.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

up_col, demo_col = st.columns([4, 1], vertical_alignment="bottom")
with up_col:
    uploaded_file = st.file_uploader(
        f"Upload {spec_info['modality']} (JPG, PNG, BMP, WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
with demo_col:
    use_demo = st.button("🧪 Try Demo Scan", use_container_width=True)

if uploaded_file is not None:
    st.session_state["active_image"] = Image.open(uploaded_file).convert("RGB")
    st.session_state["active_name"] = uploaded_file.name
elif use_demo:
    st.session_state["active_image"] = create_demo_scan(IMG_SIZE)
    st.session_state["active_name"] = "demo_phantom_scan.png"

# ======================================
# PRIMARY 2-COLUMN DIAGNOSTIC WORKSPACE
# ======================================
if "active_image" in st.session_state:
    raw_rgb = st.session_state["active_image"]
    scan_name = st.session_state.get("active_name", "uploaded_scan.png")
    orig_w, orig_h = raw_rgb.size

    enhanced_pil = enhance_scan(raw_rgb, brightness=brightness, contrast=contrast, sharpness=sharpness)
    resized_pil = enhanced_pil.resize((IMG_SIZE, IMG_SIZE))
    rgb_uint8 = np.array(resized_pil, dtype=np.uint8)
    rgb_float = rgb_uint8.astype(np.float32) / 255.0

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(enhanced_pil).unsqueeze(0).to(device)

    with st.spinner("Analyzing scan..."):
        single_probs, tta_probs = run_inference(model, input_tensor)
        probs = tta_probs if use_tta else single_probs

        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        pred_class = CLASSES[top1_idx]
        confidence = float(probs[top1_idx] * 100)
        margin = (
            float((probs[sorted_indices[0]] - probs[sorted_indices[1]]) * 100)
            if len(CLASSES) > 1
            else 100.0
        )

        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None, eigen_smooth=True)[0]

        cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, image_weight=1.0 - cam_alpha)
        contour_img, num_hotspots = draw_attention_contours(rgb_uint8, grayscale_cam, threshold=contour_thresh)

    st.write("")  # Clean breathing space

    # Clean 2-Column Split: Visual Scan on Left | Diagnosis & Actions on Right
    left_col, right_col = st.columns([1.15, 1], gap="large")

    with left_col:
        view_choice = st.segmented_control(
            "Display Mode",
            ["Side-by-Side", "Grad-CAM++ Heatmap", "Hotspot Contours", "Original Scan"],
            default="Side-by-Side",
            label_visibility="collapsed",
        )

        if view_choice == "Side-by-Side":
            img_c1, img_c2 = st.columns(2, gap="medium")
            with img_c1:
                st.image(resized_pil, caption=f"Input ({orig_w}×{orig_h} px)", use_container_width=True)
            with img_c2:
                st.image(cam_overlay, caption="Grad-CAM++ Focus", use_container_width=True)
        elif view_choice == "Grad-CAM++ Heatmap":
            st.image(cam_overlay, caption="Grad-CAM++ Localization Overlay", use_container_width=True)
        elif view_choice == "Hotspot Contours":
            st.image(
                contour_img,
                caption=f"{num_hotspots} hotspot(s) detected above {contour_thresh * 100:.0f}% attention",
                use_container_width=True,
            )
        else:
            st.image(resized_pil, caption=f"Original Scan ({orig_w}×{orig_h} px)", use_container_width=True)

    with right_col:
        # Triage Status Banner
        if confidence >= 80.0 and margin >= 25.0:
            triage_label = "High Confidence Match"
            st.success(f"**{triage_label}** (+{margin:.1f}% margin)")
        elif confidence >= 60.0 and margin >= 10.0:
            triage_label = "Moderate Confidence — Clinical Correlation Advised"
            st.info(f"**{triage_label}** (+{margin:.1f}% margin)")
        else:
            triage_label = "Low Confidence — Specialist Review Required"
            st.warning(f"**{triage_label}**")

        # Main Diagnosis Metric
        m1, m2 = st.columns(2)
        m1.metric("Predicted Condition", pred_class.upper())
        m2.metric("Confidence", f"{confidence:.2f}%")

        if pred_class.lower() in CLASS_NOTES:
            st.caption(CLASS_NOTES[pred_class.lower()])

        st.write("")
        st.markdown("**Probability Breakdown**")
        for idx in sorted_indices:
            cls = CLASSES[int(idx)]
            prob = float(probs[int(idx)])
            st.progress(prob, text=f"{cls}: {prob * 100:.2f}%")

        st.write("")
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            pdf_bytes = generate_pdf_report(
                resized_pil,
                cam_overlay,
                selected_specialty,
                scan_name,
                pred_class,
                confidence,
                triage_label,
                CLASSES,
                probs,
                arch,
                use_tta,
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"MedVision_{pred_class}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        with btn_c2:
            if st.button("➕ Save to Session Log", use_container_width=True):
                st.session_state.scan_history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "File": scan_name,
                    "Specialty": _pdf_safe(selected_specialty),
                    "Prediction": pred_class,
                    "Confidence (%)": round(confidence, 2),
                    "Triage": _pdf_safe(triage_label),
                })
                st.toast("Saved to session log!")

    st.write("")
    st.divider()

    # ======================================
    # OPTIONAL DETAILS IN CLEAN EXPANDERS
    # ======================================
    with st.expander("📊 View Single-View vs. 4-View TTA Confidence Chart", expanded=False):
        fig, ax = plt.subplots(figsize=(7, max(2.8, len(CLASSES) * 0.48)))
        y_pos = np.arange(len(CLASSES))
        ordered_classes = [CLASSES[int(i)] for i in sorted_indices][::-1]
        ordered_single = [single_probs[int(i)] * 100 for i in sorted_indices][::-1]
        ordered_tta = [tta_probs[int(i)] * 100 for i in sorted_indices][::-1]

        ax.barh(y_pos - 0.18, ordered_single, height=0.35, label="Single-View", color="#94a3b8")
        ax.barh(y_pos + 0.18, ordered_tta, height=0.35, label="4-View TTA", color="#0284c7")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ordered_classes)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Confidence (%)")
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander(f"📋 Session Diagnostic Log ({len(st.session_state.scan_history)} saved)", expanded=False):
        if st.session_state.scan_history:
            st.dataframe(st.session_state.scan_history, use_container_width=True)
            l1, l2 = st.columns([1, 1])
            with l1:
                st.download_button(
                    label="Download CSV Log",
                    data=history_to_csv(st.session_state.scan_history),
                    file_name=f"MedVision_Log_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with l2:
                if st.button("Clear Log", use_container_width=True):
                    st.session_state.scan_history = []
                    st.rerun()
        else:
            st.caption("No scans saved yet. Click **Save to Session Log** after analyzing a scan.")

else:
    st.info(
        f"Upload a **{spec_info['modality']}** above or click **🧪 Try Demo Scan** to begin analysis."
    )