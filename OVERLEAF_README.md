## Overleaf 兼容版使用说明

已准备好自动打包脚本：

- `prepare_overleaf_package.py`

### 1) 生成上传目录

在项目根目录运行：

```bash
cd /data2/wjb/code/PTMC_LNM
python prepare_overleaf_package.py
```

运行后会生成：

- `overleaf_package/main.tex`
- `overleaf_package/figures/*.png`
- `overleaf_package/README_OVERLEAF.txt`

### 2) 上传到 Overleaf

1. 新建 Overleaf 项目（Blank Project）。
2. 把 `overleaf_package/` 里的所有文件上传到项目根目录（保持 `figures/` 子目录）。
3. 在 Overleaf 菜单里把编译器设置为 **XeLaTeX**。
4. 编译 `main.tex`。

### 3) 说明

- 该版本已经使用相对路径引用图片（`figures/...`），可直接在 Overleaf 编译。
- 后续若你更新图表，重新运行打包脚本后再覆盖上传即可。

