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
python main.py scan.tif                          # notch 去网纹
python main.py scan_descreen.tif --blur          # 高斯模糊
python main.py scan_descreen_blur.tif --noise    # 蒙尘与划痕
python main.py ..._blur_noise.tif --sharpen      # USM 锐化
```
## Examples

```sh
python main.py scan.tif -o out.tif --spectrum    # 指定输出路径, 保存频谱图
```
