import os
from PIL import Image, ImageOps, ImageDraw


def add_title(img: Image.Image, title: str, h: int = 38) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + h), "white")
    canvas.paste(img, (0, h))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="black")
    return canvas


def main():
    base = "./checkpoints/exp_fullmodal_cascaded_clinical_bs16_e150"
    case = "41497470_1_3_1"

    gradcam_path = os.path.join(base, "gradcam_vis", f"{case}_task-LateralLNM.png")
    module_path = os.path.join(base, "module_contrib", f"case-{case}_task-lateral_modules.png")
    out_path = "../figures/case_gradcam_module_panel.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(gradcam_path):
        raise FileNotFoundError(f"missing gradcam image: {gradcam_path}")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"missing module image: {module_path}")

    left = Image.open(gradcam_path).convert("RGB")
    right = Image.open(module_path).convert("RGB")

    target_h = max(left.height, right.height)
    left = ImageOps.pad(left, (int(left.width * target_h / left.height), target_h), color="white")
    right = ImageOps.pad(right, (int(right.width * target_h / right.height), target_h), color="white")

    left = add_title(left, "Grad-CAM (LateralLNM)")
    right = add_title(right, "Module Contribution (LateralLNM)")

    gap = 20
    panel = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height)), "white")
    panel.paste(left, (0, 0))
    panel.paste(right, (left.width + gap, 0))
    panel.save(out_path, quality=95)
    print(f"saved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()

