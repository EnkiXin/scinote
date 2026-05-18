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

## 7. C0 video-only 结果（Qwen2.5-VL-7B，H200，统一 32 帧，2026-05-16）

最终全量数据来自 [PROGRESS.md](PROGRESS.md) 的 ExpVid headline 表。Paper 列是 ExpVid 论文中 **QwenVL2.5-7B**（不是 78B）的对应数字，做复现对照用。

| Task | n | Ours (Video) | Paper 7B |
|------|---:|---:|---:|
| L1 materials                | 1266 | 34.04 | 33.9 |
| L1 tools                    | 1130 | 36.28 | 32.0 |
| L1 operation                | 938  | 64.61 | 62.4 |
| L1 quantity                 | 701  | 47.22 | 49.0 |
| **L1 avg**                  | 4035 | **45.54** | **42.6** (+2.9) |
| L2 sequence_generation (F1) | 750  | 43.32 | 20.8 (Jaccard) |
| L2 sequence_ordering        | 739  | 52.64 | 56.2 |
| L2 step_prediction          | 748  |  2.14 |  1.3 |
| L2 video_verification       | 748  | 17.38 | 20.7 |
| **L2 avg**                  | 2985 | **28.87** | **24.6** (+4.3) |
| L3 experimental_conclusion  | 390  | 21.28 | 25.2 |
| L3 scientific_discovery     | 390  | 20.00 | 21.4 |
| **L3 avg**                  | 780  | **20.64** | **23.3** (−2.7) |

**复现结论**：与论文 Qwen2.5-VL-7B 数字在 +3 pp 以内（L1/L2 略高、L3 略低）。L2 sequence_generation 指标不同（我们用 F1，论文用 Jaccard），数值不可直接比。

Notes & oracle 条件的全部数据见 [PROGRESS.md](PROGRESS.md)。
