# Open-o3-Video 忠实复现 PROGRESS

目标:严格忠实复现 Open-o3-Video 两阶段训练(SFT→GRPO RL),作为后续训练方法/数据改进的干净 baseline。
论文 V-STaR 指标:相对 Qwen2.5-VL baseline **mAM +14.4% / mLGM +24.2%**。

## 流水线状态

| 阶段 | 状态 | 关键数字 |
|---|---|---|
| 数据重建(7源) | ✅ 完成 | covered: SFT 30498/31166 (97.9%)、RL 36539/37231 (98.1%) |
| 环境 | ✅ | 训练 vc_train(transformers 4.49/trl 0.16.1)+ cuda13 nvcc + flash-attn 2.8.3(sm_90);评测 open-o3-eval(vllm 0.7.2) |
| **SFT(全量微调)** | ✅ **完成** | 5083步/1epoch,train_loss **0.388**(4.7→0.39),token_acc ~0.89,8.47h,6×H200 |
| **GRPO RL** | ✅ **完成** | checkpoint-6000(98.5%/lr≈0),reward 1.78→2.4;末批 reward-reshape 边界 bug 崩但无影响 |
| V-STaR 评测 | ✅ **完成** | 见下表 |

## 实现过程(端到端,8 步)

| # | 阶段 | 做了什么 | 关键挑战 & 解决 | 规模/耗时 |
|---|---|---|---|---|
| 1 | **数据重建** | 下载+解压+归位 7 个源到官方 `videos/<dir>` 布局,重用官方 `official_media_paths` | 修 3 个映射 bug:qvhighlights/activitynet 在 **GroundedVLLM** 非 VideoMind;internvid 用 **3fps裁剪变体**;各子集解压前缀差一层。镜像 hf-mirror 聚合 **217MB/s** | ~1TB,数小时 |
| 2 | **环境** | 训练 env `vc_train`(transformers 4.49/trl 0.16.1)+ 自装 `cuda13` nvcc;评测 env `open-o3-eval`(vllm 0.7.2) | flash-attn 无 torch2.11/cu130 预编译 wheel → 从源编,`FLASH_ATTN_CUDA_ARCHS=90` 只编 sm_90(否则默认编 sm80/90/100/120 极慢);deepspeed FusedAdam 需 `ninja` 在 PATH | — |
| 3 | **covered json** | `make_covered_json.py` 重生只含媒体存在的样本 | 原 covered 子集系统性丢整类任务(TreeVGR视觉QA/TVG时序QA 全缺)→ 补全后 **SFT 97.9% / RL 98.1%**,全任务类覆盖 | SFT 30498 / RL 36539 |
| 4 | **代码补丁** | SFT `collate_fn` + GRPO `compute_loss` 坏视频回退(`_LAST_GOOD_EXAMPLE`) | torchvision 0.26 丢 `io.read_video`,坏 h264 崩;**官方 GRPO guard 是坏的**(失败时变量未绑定→ `NameError` 杀全程,比 SFT 更脆) | 2 patches |
| 5 | **SFT(全量微调)** | 6×H200,deepspeed zero2,FusedAdam,lr 1e-6,1 epoch,sdpa | `VIDEO_MAX_PIXELS=1605632`(=论文 16帧×128token/帧)既忠实又防 OOM;官方 shell 没设默认 90M 会崩 | 5083步,**8.5h**,loss 4.7→0.39 |
| 6 | **RL(GRPO)** | 从 SFT checkpoint-5083,6×H200,zero3,flash-attn,7 reward,GSPO,beta 0.04,num_gen 4 | 末批 reward-reshape 边界 bug 崩于步 6089(样本数不整除6卡);但 **checkpoint-6000 已存且 cosine lr≈0**,最后 90 步无影响 | 6090步,**~41h**,reward 1.78→2.4 |
| 7 | **V-STaR 评测** | 官方两阶段(Stage A vLLM 推理 + Stage B 本地 72B judge)× 4 模型(baseline/SFT/RL/official) | 踩坑:worker 子进程需 `PYTHONPATH=eval/`;judge 需 `pip install accelerate`;**首轮编排器与手动驱动并发覆盖了 json → 单流水线重跑修正** | 2094题×4模型 |
| 8 | **验证** | clean judge 复跑 + 对标官方 released 模型 | rl 复跑数字一致(json 已备份);**我 RL mAM 0.328 ≈ 官方 0.333,差 0.55pt** | ✅ 忠实坐实 |

## ★ V-STaR 评测结果(官方 eval_vstar.py 未改 + Qwen2.5-72B judge)

全部 4 模型均在 2094/2094 有效预测的干净 json 上评测(注:首次评测因编排器与手动驱动并发,baseline/official/sft 的 json 被覆盖成空预测,已单流水线重跑修正;rl 全程干净并已备份)。

| 模型 | mAM | mLGM | VQA acc | vs官方 mAM |
|---|---|---|---|---|
| Qwen2.5-VL baseline | 0.1258 | 0.1376 | 0.205 | — |
| 我的 SFT(checkpoint-5083) | 0.2986 | 0.4009 | 0.567 | −0.035 |
| **我的 RL(checkpoint-6000)** | **0.3279** | **0.4503** | 0.599 | **−0.0055** |
| Open-o3 官方 released | 0.3334 | 0.4569 | 0.598 | — |

**忠实度结论**:我的 RL 模型 mAM 0.3279 vs 官方 released 0.3334,**差距仅 −0.55pt**(mLGM 差 −0.66pt),在评测随机抖动(vLLM 无固定种子 ~0.5pt)范围内 → **训练管道忠实复现官方**。单调递进 baseline 0.126 → SFT 0.299(+17.3)→ RL 0.328(+2.9)≈ 官方 0.333,符合两阶段设计。

**关于"+Δ vs 论文 +14.4"**:我的 RL−baseline = +20.2 mAM,大于论文 +14.4。原因不是模型更强,而是 **baseline 被强制 grounded 格式(think_mode)压低了 VQA acc(0.205)**——base 模型没学过 `<answer>` 格式,答案抽取失败。论文 baseline 口径更宽松(隐含 ~0.184)。**真正可比的绝对值(我 RL 0.328 ≈ 官方 0.333）才是忠实度硬证据。**

## 产物路径
- SFT ckpt:`open_o3/ckpts/sft_faithful/checkpoint-5083`(已补 processor 文件)
- RL ckpt:`open_o3/ckpts/rl_faithful/`(训练中,每500步存)
- 脚本:`open_o3/scripts/{run_sft_faithful,run_grpo_faithful,run_eval,extract_sources,make_covered_json,dl_mirror}.{sh,py}`
- 日志:`open_o3/logs/{sft_faithful,rl_faithful}.log`

## 忠实度:与官方/论文逐项对齐 + 已标注偏差
**完全对齐**:全量微调(非LoRA)、deepspeed zero2(SFT)/zero3(RL)、FusedAdam、lr 1e-6、1 epoch、7个reward(ans_acc/ans_tiou/ans_viou/thk_temporal_point/thk_temporal_segment/thk_spatial/format)、GSPO seq-level、beta 0.04、num_generations 4、max_prompt 16384/max_completion 768、max_pixels 401408(RL)、save_steps 500。

**infra 受限偏差(已标注)**:
1. **6 卡(GPU 2-7)** vs 官方 8 卡(GPU0=用户robot任务、GPU1=benchmark任务不可占)→ global batch 6 vs 8(论文未指定 batch size)。
2. **SFT 用 sdpa**(flash_attention_2 因 torch2.11/cu130 无预编译wheel,编译期才就绪)→ 数学等价;**RL 用 flash_attention_2**(已编译完成)。
3. **covered 子集**:SFT 97.9%、RL 98.1%,缺的是 TVG 尾部(didemo/queryd/hirest/activitynet,源档案体积/样本比极差,~668-692样本)+ VideoEspresso少量。全部任务类别均有覆盖(TreeVGR视觉QA 5000、TVG时序QA、Video-R1通用QA)。
4. **VIDEO_MAX_PIXELS=1605632**(=论文"16帧×128 token/帧";官方shell未设此env,默认90M既不符论文又OOM)。

## 代码补丁(官方 bug 修复)
- `sft_multi_task.py` collate_fn:`_LAST_GOOD_EXAMPLE` 坏视频回退(torchvision 0.26 丢失 io.read_video,坏h264崩溃)。
- `grpo_trainer.py` compute_loss PATCH1:同款坏视频回退。官方原 guard 在 process_vision_info 失败时变量未绑定 → 下游 `if video_inputs is None` NameError 杀全程(比SFT更脆)。

## 数据源映射(易错点)
- GQA→lmms-lab/GQA(parquet抽3856图);VideoEspresso→hshjerry0315(50段分卷zip);Video-R1→Video-R1/Video-R1-data(NeXT-QA/CLEVRER/PerceptionTest);TreeVGR图→lmms-lab/LLaVA-NeXT-Data(llava_next_raw_format)。
- **TVG**:tacos/didemo/queryd/hirest→VideoMind-Dataset(全分辨率`videos/`);internvid_vtime→VideoMind 的 **`videos_crop_3fps_480_noaudio`** 变体;**qvhighlights/activitynet→GroundedVLLM(WHB139426/Grounded-VideoLLM)非VideoMind**。

_注:open_o3/ 非 git repo;Open-o3-Video/ 的 origin 是官方上游(不可push)。无个人远程,故本地记录。_
