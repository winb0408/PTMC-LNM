#!/usr/bin/env python3
"""
prepare_overleaf_package.py

用途：
- 生成一个 Overleaf 兼容上传目录 overleaf_package/
- 自动复制主 tex 文件和当前报告依赖图片

运行：
  cd /data2/wjb/code/PTMC_LNM
  python prepare_overleaf_package.py
"""

import os
import shutil
from pathlib import Path


ROOT = Path("/data2/wjb/code/PTMC_LNM")
OUT = ROOT / "overleaf_package"
OUT_FIG = OUT / "figures"

MAIN_TEX = ROOT / "CentralLNM_Analysis_Report.tex"

FIG_FILES = [
    ROOT / "figures" / "method_flow_diagram.png",
    ROOT / "figures" / "auc_boxplot_three_routes.png",
    ROOT / "figures" / "roc_overlay_imageonly_b.png",
    ROOT / "figures" / "roc_overlay_three_routes.png",
    ROOT / "figures" / "case_gradcam_module_panel.png",
]


def copy_file(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"missing file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    # 复制主 tex
    copy_file(MAIN_TEX, OUT / "main.tex")

    # 复制图片
    for p in FIG_FILES:
        copy_file(p, OUT_FIG / p.name)

    # 写一个最小说明
    readme = OUT / "README_OVERLEAF.txt"
    readme.write_text(
        "Overleaf usage:\n"
        "1) Upload all files in this folder to Overleaf.\n"
        "2) Set compiler to XeLaTeX.\n"
        "3) Compile main.tex.\n",
        encoding="utf-8",
    )

    print(f"[OK] Overleaf package ready: {OUT}")
    print("Main file: overleaf_package/main.tex")
    print("Please set Overleaf compiler to XeLaTeX.")


if __name__ == "__main__":
    main()

