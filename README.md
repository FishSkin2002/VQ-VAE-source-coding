# 项目说明

本项目包含三个主要任务：
- 训练 VQ-VAE/VQ-VAE-2（train.py）
- 使用预训练 VQ-VAE-2 的嵌入训练分类器（classify.py）
- 端到端链路仿真与可视化（simulate/main.py）

目录结构：
```
.
├─train.py            # 训练 VQ-VAE / VQ-VAE-2
├─classify.py         # 训练分类器
├─simulate/
│  ├─main.py          # 链路仿真与可视化，输出 PNG/CSV/比特流
│  ├─link_sim.py      # 调制/OFDM/信道/重复码 仿真
│  └─vqvae2_codec.py  # VQ-VAE-2模型构建、编码/解码工具
├─model/
│  ├─vqvae.py         # VQ-VAE模型定义
│  ├─vqvae2.py        # VQ-VAE-2模型定义
│  └─classifier.py    # 全连接分类器
├─configs/
│  └─train.yaml       # 训练配置
├─runs/
│  ├─exp/             # 预训练VQ-VAE-2
│  └─classifier/      # 预训练分类器
└─dataset/            # 数据集保存位置
```

## 环境配置
本项目使用 Python 3.10，依赖安装：
```
pip install -r requirements.txt
```

## 训练 VQ-VAE / VQ-VAE-2（train.py）
使用配置文件 `configs/train.yaml`，可中断后继续训练，示例：
```
python train.py --config configs/train.yaml --resume exp 200
```
参数：
- `--config`：配置文件路径（默认 configs/train.yaml）
- `--resume`：在 runs/ 目录下指定要继续训练的实验（例如 exp），可输入第二个整数参数代表修改总epoch（如第一次训练设置100 epoch，可改为200 epoch）

## 训练分类器（classify.py）
使用预训练 VQ-VAE-2（默认 runs/exp），提取 decoder 输入的量化嵌入训练分类器，VQ-VAE-2不会被训练：
```
python classify.py --epochs 30 --batch-size 256 --amp
```
主要参数：
- `--epochs`：默认 30
- `--batch-size`：默认 256
- `--lr`：学习率
- `--min-lr`：余弦学习率调度最小学习率
- `--amp`：混合精度（默认开启）
- `--device`：cuda/cpu 自动
- `--save-path`：默认 runs/classifier/best.pt（验证集指标最优时覆盖）

## 仿真与可视化（simulate/main.py）
生成一张左侧大原图，右侧 3 列信道 × 多行 SNR 的网格，并保存 PNG/CSV/比特流：
```
python simulate/main.py -p 0
```
主要参数：
- `-p/--pic`：CIFAR-10 测试集索引（默认 0）
行为：
- 默认使用 runs/exp 的 best.pt 与 config.yaml
- SNR 列表：在代码内 `snr_list = [15, 10, 5]`（可自行调整）
- 输出：simulate/outputs/mode_\<mode\>.*

## 常见问题
- 运行前确保 data/cifar-10-batches-py 已准备好，或允许 torchvision 自动下载。
- 训练/仿真均默认使用 runs/exp 和 runs/classifier 下的模型，如需切换请更新 vqvae2_codec.py 的默认路径或传参。
- 分类器仅更新自身参数，须确保已有训练好的 VQ-VAE-2 模型。
