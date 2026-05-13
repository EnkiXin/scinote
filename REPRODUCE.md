# ExpVid 评测复现指南

## 目标

在 ExpVid benchmark 上评测 Qwen2.5-VL-7B-Instruct，复现论文结果。

---

## 1. 服务器准备（RTX 4090 推荐）

连接 featurize.cn 服务器，按以下步骤初始化：

```bash
bash setup_server.sh
```

`setup_server.sh` 安装：
- PyTorch (CUDA 12.1)
- transformers==4.49.0（**必须 <5.0，否则与 PyTorch 2.2 不兼容**）
- qwen-vl-utils, av, pillow, numpy, huggingface_hub

---

## 2. 上传代码

从本地 Mac 上传到服务器：

```bash
scp -P 26795 evaluate.py featurize@workspace.featurize.cn:~/expvid/
# 或用 paramiko SFTP 脚本（服务器不支持密码 scp 时）
```

---

## 3. 运行评测

```bash
cd ~/expvid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 防止显存碎片化 OOM

# 全量跑所有任务（后台运行，断线不中断）
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 evaluate.py --task all --output results --resume \
    >> eval_full_run.log 2>&1 &
echo "PID: $!"
```

- `--resume`：跳过已有结果的任务，崩溃后可直接重启
- `--limit N`：每个任务只跑前 N 条，用于快速测试
- `--task materials`：只跑单个任务

---

## 4. 监控进度

```bash
tail -f ~/expvid/eval_full_run.log
nvidia-smi  # 查看显存
```

---

## 5. 任务类型说明

| 任务 | 类型 | 评分方式 |
|------|------|---------|
| L1: materials, tools, operation, quantity | mc | 精确匹配 A/B/C/D |
| L2: sequence_ordering, video_verification | mc | 精确匹配 A/B/C/D |
| L2: sequence_generation | seqgen | 步骤编号集合 F1 |
| L2: step_prediction | steppred | 步骤编号精确匹配 |
| L3: experimental_conclusion, scientific_discovery | fitb | token-level F1 |

---

## 6. 已知问题与修复

| 问题 | 原因 | 修复 |
|------|------|------|
| `'list' object has no attribute 'to'` | processor 返回混合字典 | `v.to() if hasattr(v, 'to')` |
| CUDA OOM | 长视频帧占用超出空闲显存 | `torch.cuda.empty_cache()` + `expandable_segments=True` |
| `KeyError: 'options'` | L2 部分任务无选项字段 | 新增 seqgen/steppred 任务类型 |
| transformers 5.x 不兼容 | 需要 PyTorch ≥ 2.4 | 固定 `transformers==4.49.0` |

---

## 7. 结果汇总（Qwen2.5-VL-7B，2026-05-11）

| Task | Accuracy | 论文 QwenVL2.5-78B |
|------|----------|-------------------|
| materials | 34.28% | — |
| tools | 35.66% | — |
| operation | 64.43% | — |
| quantity | 48.78% | — |
| **L1 平均** | **~45.8%** | **43.9%** |
| sequence_generation | 🔄 进行中 | — |
| sequence_ordering | ⏳ | — |
| step_prediction | ⏳ | — |
| video_verification | ⏳ | — |
| **L2 平均** | **⏳** | **35.9%** |
| experimental_conclusion | ⏳ | — |
| scientific_discovery | ⏳ | — |
| **L3 平均** | **⏳** | **30.6%** |

持续更新见 [PROGRESS.md](PROGRESS.md)。
