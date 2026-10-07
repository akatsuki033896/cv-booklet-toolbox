**The toolbox's default settings is for 600-dpi.**

## Quick Start

create environment by `env.yaml`

```sh
conda env create -f env.yaml
conda activate cv
```

processing

```sh
python main.py scan.tif --all                    # 一条命令完成全部四步
```
## Examples

Processing

```sh
python main.py scan.tif                          # notch 去网纹
python main.py scan_descreen.tif --blur          # 高斯模糊
python main.py scan_descreen_blur.tif --noise    # 蒙尘与划痕
python main.py ..._blur_noise.tif --sharpen      # USM 锐化
```

Output to existed path

```sh
python main.py scan.tif --all -o output/scan.tiff
```

View the spectrum

```sh
python main.py scan.tif -o out.tif --spectrum    # 指定输出路径, 保存频谱图
```
Rename output files

```sh
python batch_rename.py <path> --dry-run          # 预览: 批量删除文件名第一个 _ 及其后部分
python batch_rename.py <目录>             # 实际执行
python batch_rename.py <path> [-r]               # 执行重命名(-r 递归子目录), 目标已存在则跳过
```
