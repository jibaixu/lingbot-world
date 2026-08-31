# Confidence-Aware 特征抽取实现说明

本文档对应 `/home/jbx/Codes/starVLA_origin/starVLA/confidence_aware_teacher_distillation.md`，说明本仓库已经落地的 LingBot spatial teacher 抽取部分，以及交给顶层蒸馏项目消费的缓存契约。

## 已实现内容

`tools/extract_lingbot_latents.py` 保留两条路径：

- base 路径仍然是每个 anchor 一次 diffusion 推理，只写 `features` 和 `anchor_frames`；不带新参数时行为不变。
- confidence-aware 路径使用 `--confidence-aware --num-teacher-samples 4`。同一 anchor 的四次推理复用同一张图像、同一条确定性相机轨迹、同一 prompt、几何、layer 和 denoising steps，仅改变 diffusion noise seed。

每次结果会立即转为 CPU 临时 tensor。四次结果聚合后只保留均值 feature、raw disagreement 和 confidence；临时四份样本随当前 anchor 结束释放，不写入长期缓存。

disagreement 的计算为每个 token 做最后一维 LayerNorm 和 L2 normalization，再对所有样本对计算 cosine disagreement，最后对 token 取 median。均值 target 使用 float32 聚合，写盘时恢复现有 `bfloat16` feature dtype。

## 校准与文件协议

confidence-aware 抽取先为每个 episode 写入 `complete=false` 的临时协议文件，其中 confidence 是占位值，disagreement 已经是真实值。所有选定 train episodes 完成后，抽取器收集 disagreement 标量，计算 q05/q95，并原子重写每个 episode 文件：

```text
features       [A, N, D]   bfloat16
anchor_frames  [A]         int64
confidence     [A]         float32
disagreement   [A]         float32
```

校准公式为 `1 - clip((u-q05)/(q95-q05+eps), 0, 1)`。校准参数写入输出目录的 `confidence_manifest.json`，并写入每个 episode 的 metadata。只有校准完成后 episode 才会变为 `complete=true`。

confidence-aware 文件至少包含以下 metadata：

```text
confidence_aware=true
confidence_version=v1
num_teacher_samples=4
target_aggregation=mean
uncertainty_metric=pairwise_cosine_disagreement
feature_normalization=layernorm_last_dim
uncertainty_aggregation=median_token
calibration=train_quantile_05_95
seed_policy=explicit_per_anchor_fixed_trajectory
```

## 运行方式

`run_extract_lingbot_latents.sh` 默认执行四次采样版本，并保留注释掉的 base 命令用于 legacy/equal-weight 对照。示例输出目录为：

```text
lingbot_latent_step4_numlayer24_stride16_samples4_confv1
```

生成训练 episode 索引时使用：

```bash
python tools/update_episodes_train_lingbot.py \
  --data-root /data/jibaixu/Datasets/LIBERO/libero_lerobot_v2.1 \
  --latent-dir lingbot_latent_step4_numlayer24_stride16_samples4_confv1 \
  --require-confidence \
  --overwrite
```

`--require-confidence` 会严格检查 finalized metadata、confidence/disagreement shape 和 anchor 数量；不带该参数时保持旧 validator 的兼容行为。

## 与顶层蒸馏项目的边界

本仓库负责生成均值 teacher target、raw disagreement、校准 confidence 及其 metadata。starVLA 训练端需要：

1. loader 在 confidence-aware 模式读取 `confidence[a]`，旧缓存只允许 legacy 模式；
2. packed cache 将 confidence 与 features 分开 mmap；
3. alignment 返回 per-example loss，并按 `w_min + (1-w_min)*confidence` 做有效权重归一化；
4. confidence 不输入 AISER，仅影响 alignment supervision。

这些训练期行为不在 LingBot-World 推理代码中执行，部署路径也不会读取 teacher cache 或 confidence。

## 验证状态

已覆盖静态语法检查、合成 tensor 的 disagreement 计算、confidence 范围和缓存 shape/metadata 校验。真实模型抽取需要 CUDA、checkpoint 和 LIBERO 数据集，未在当前环境中运行完整八卡任务。
