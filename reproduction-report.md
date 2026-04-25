# 复现对齐报告

复核日期：2026-04-24

本报告将当前仓库与论文 *A Recipe for Generating 3D Worlds From a Single Image* 进行再次全面对比。复核方式仅限静态代码审阅和论文文本核对，未运行 diffusion、VLM、depth、3DGS 或 VR 相关流程。

## 依据来源

- 论文 PDF：[A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025.pdf](A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025.pdf)
- arXiv 页面：<https://arxiv.org/abs/2503.16611>
- 项目页面：<https://katjaschwarz.github.io/worlds/>
- 当前实现：[pipeline.py](pipeline.py)、[pipeline_helper.py](pipeline_helper.py)、[pipeline.toml](pipeline.toml)
- 既有本地笔记：[reproduction-concept.md](reproduction-concept.md)、[reproduction-plan.md](reproduction-plan.md)

## 总体结论

当前仓库更准确的定位是：论文 Section 3.1 panorama generation 的部分复现。它实现了 anchored panorama synthesis 的主流程、核心 perspective/equirectangular 几何、VLM 方向性 prompt 拆分、基于公开 Diffusers 的 inpainting 后端，以及可选 refinement。

当前仓库不是完整论文复现。论文 Section 3.2 和 Section 3.3 中把全景图提升为可导航 3D world 的部分，包括 metric depth、point-cloud-conditioned inpainting、3D Gaussian Splatting 和 VR 展示，均未实现。

## 完全对齐部分

以下部分在静态实现层面与论文算法描述一致。

| 论文要求 | 当前实现 | 证据 |
|---|---|---|
| 单张输入图嵌入 2:1 equirectangular panorama。 | 管线加载单张 RGB 图像，并创建 2:1 全景画布。 | `ImageIO.load_image`、`GeometryTools.create_equirectangular_canvas` |
| 具备 perspective 到 equirectangular，以及 equirectangular 到 perspective 的几何转换。 | 实现了输入图投影到全景图，以及从全景图渲染 perspective crop。 | `project_perspective_to_equirect`、`render_perspective_from_equirect` |
| 在给定水平 FoV 后，按等焦距假设推导垂直 FoV。 | `estimate_fov` 用 `fov_x`、图像宽高推导 `fov_y`。 | `GeometryTools.estimate_fov` |
| anchored synthesis 的主顺序。 | 输入图投影到正面，复制到 yaw 180 作为临时背面 anchor，先生成 front/back top/bottom，删除 anchor，再生成 horizontal views，最后生成 side top/bottom。 | `AnchoredSynthesizer.initialize`、`remove_backside_anchor`、`run` |
| 默认视角数量和 FoV 与论文 Section 3.1 主 recipe 一致。 | schedule 使用 4 个斜向 top view、4 个斜向 bottom view、8 个 horizontal view；顺序为 front/back vertical、horizontal、side vertical；默认 top/bottom FoV 为 120 度，pitch 为 `+/- 45`，middle band 为 85 度。 | `ViewSchedule.anchored`、`pipeline.toml` |
| direction-specific prompt 拆分。 | VLM schema 包含 scene type、global atmosphere、sky/ceiling、ground/floor、negative prompt；top/bottom view 会组合区域 prompt 和 global prompt。 | `PromptTools`、`prompt_for_view` |
| prompt comparison modes。 | 默认使用论文效果最好的 directional prompts，同时支持 `coarse` 和 `caption` 对比模式。 | `[prompting] mode`、`PromptTools.effective_prompts_for_mode` |
| main synthesis mask edge cleanup。 | raw binary mask 先膨胀，再生成 soft mask；binary mask 保留用于几何与 known-mask 判断，soft mask 传给 Diffusers 并用于 soft masked view。 | `GeometryTools.dilate_mask`、`GeometryTools.blur_mask`、`soft_inpaint_mask` |
| backside anchor 不作为最终内容保留。 | 进入 horizontal 阶段前清除 anchor-only 像素，并保留原始 input mask。 | `remove_backside_anchor` |
| 原始输入区域在生成与 refinement 中被保护。 | panorama 更新和 overlap blending 都排除受保护输入区域，refinement 投影时也排除 `input_mask`。 | `PanoramaUpdater.update_with_view`、`PanoramaRefiner.project_refined_view` |
| 保存中间调试产物。 | prompts、masks、rendered views、inpainted views、projected updates、records、final panorama 等写入 `outputs/<timestamp>/debug`。 | `DebugWriter`、`pipeline.py` |

## 部分实现但未完全对齐

以下部分符合论文思路，但模型、默认参数或低层行为与论文不同。

| 主题 | 论文 | 当前实现 | 差异 |
|---|---|---|---|
| 水平 FoV 估计 | 使用 Dust3R 估计 `fov_x`。 | 读取配置项 `panorama.input_fov_x`，默认 `70.0`。 | 未实现 Dust3R FoV 估计。 |
| inpainting 模型 | 使用 proprietary transformer-based T2I diffusion model，并通过 ControlNet 消化 masked input。 | 使用公开 Diffusers 后端，默认 `stabilityai/stable-diffusion-2-inpainting`。 | 是公开替代模型，不是论文模型，也没有匹配论文 ControlNet 架构。 |
| view resolution | 论文使用 1024 px side length 的 square inpainting view。 | 默认 `view.size = 512`，代码可通过配置改大。 | 默认低于论文设置。 |
| panorama resolution | 论文高分辨率全景评估使用 2:1 panorama，表格中以 2048 x 4096 像素记法呈现；按本仓库 W x H 约等于 4096 x 2048。 | 默认 2048 x 1024。 | 默认每个维度为论文尺度的一半。 |
| prompt model | 使用 Llama 3.2 Vision 生成方向性 prompt，并用 Florence-2 作为 caption baseline。 | 使用任意 OpenAI-compatible VLM endpoint，默认本地 `qwen3.6`；`caption` mode 也使用该 VLM。 | prompt behavior 对齐；Florence-2 不实现，因为它不是论文最终方法。 |
| prompt ablation | 比较 caption prompt、coarse prompt、directional prompts。 | `prompting.mode` 支持 `caption`、`coarse`、`directional`。 | 支持单模式对比，但未实现一个命令同时跑三路输出。 |
| inpainting mask 边缘平滑 | 附录说明 projection/inpainting 时加入轻微 blur，以避免 sharp mask edges。 | main synthesis 从 raw binary mask 生成 dilated `inpaint_mask`，再生成 `soft_inpaint_mask` 并传给 Diffusers。 | dilation 和 blur 参数可配置，不是论文精确参数。 |
| overlap blending | 论文没有完整说明 overlapping generated views 的融合细节。 | 对 already-generated 且非 input/anchor 区域支持可选保守 image-space overlap blending。 | 融合策略是工程补全，论文未充分指定。 |
| refinement | 论文使用 standard T2I diffusion，对最后 30% timesteps 做 partial denoising，并用 blurred inpainting mask blend。 | 使用 Diffusers img2img，`denoise_strength = 0.3`，按 schedule 的 perspective view 分块 refinement，再用 blurred mask blend。 | 对齐 30% denoising 思路，但 full panorama 还是 per-view refinement 的精确协议无法确认。 |
| seed policy | 论文没有充分公开每个 view 的 seed 细节。 | synthesis 从 `seed` 创建一个共享 generator；refinement 从 `seed + 1000` 创建一个共享 generator；每次调用自然推进随机流。 | 可复现并避免每个 view 重复随机起点，但仍不是论文明确公开的低层策略。 |

## 未实现部分

以下论文组件当前仓库中不存在。

### Panorama Generation 与 Ablation

- Dust3R-based `fov_x` estimation。
- ad-hoc panorama synthesis baseline。
- sequential synthesis baseline。
- Florence-2 caption baseline 不实现；caption 对比模式改用当前配置的 VLM endpoint。
- non-specific prompt baseline。
- BRISQUE、NIQE、Q-Align、CLIP-I 的 quantitative panorama evaluation。
- 用 6 个 360-degree rendered views 进行 panorama evaluation 的协议。
- 论文尺度默认配置：1024 px inpainting views、W x H 约 4096 x 2048 panoramas。

### Point Cloud-Conditioned Inpainting，论文 Section 3.2

- 从生成的 panorama 渲染 perspective images 用于 depth prediction。
- Metric3Dv2 depth estimation。
- MoGE depth estimation。
- 用 Metric3Dv2 对 MoGE depth 做 quantile-based scale alignment。
- ground average distance 至少约 1.5 m 的约束。
- panorama-to-point-cloud construction。
- 支持带 translation 的可导航 camera poses。
- 从 translated camera pose 渲染 point cloud。
- 从 point cloud render 得到 occlusion masks。
- 基于 CUT3R 的 on-the-fly camera pose 和 point cloud training data。
- forward-warp training strategy。
- forward-backward-warp training strategy。
- point-cloud-conditioned ControlNet fine-tuning 5k iterations。
- 2 m cube setup 中的 14 个 translation vectors。
- 每个 translation 的 14 个 rotations。
- 基于 rendered point-cloud condition 的 novel-view inpainting。

### 3D Reconstruction，论文 Section 3.3

- 3D Gaussian Splatting reconstruction。
- NerfStudio Splatfacto integration。
- 用 lifted panorama point cloud 初始化 splats。
- 5k-step shortened Splatfacto training schedule。
- disabled periodic opacity reset。
- adaptive density-control schedule 调整。
- spherical harmonic degree setting。
- GS batch-size change。
- point-cloud-conditioned images 只使用 inpainted regions 训练。
- panorama-synthesis images 排除 backside anchor areas。
- trainable image distortion model。
- per-image distortion embeddings。
- harmonic position embeddings for distortion。
- 128 x 128 low-resolution distortion grid 与 bilinear upsampling。
- 可实时 VR 展示的 3DGS artifact。

### Experiments 与 Comparisons

- Tanks and Temples Advanced dataset selection workflow。
- World Labs input-image comparison setup。
- DL3DV-10K ControlNet training setup。
- ScanNet++ evaluation setup。
- DiffusionLight、MVDiffusion、Diffusion360 comparison runs。
- WonderJourney、DimensionX comparison runs。
- 用 CUT3R 为 baselines 提取 poses。
- 3DGS image-quality evaluation trajectories。
- Metric3D 与 MoGE depth estimator ablation。
- text-to-world extension 和 DreamScene360 comparison。
- failure-case reproduction、resampling 或 user prompt edit workflow。

## 不确定部分

以下内容无法仅凭静态审阅确认，或论文自身未充分公开低层细节。

- projection orientation 是否完全正确：公式和函数存在，但 yaw/pitch 符号、极点方向需要视觉 round-trip 检查。
- `DiffusionPipeline` 与当前 inpainting checkpoint 的运行兼容性，包括 `image`、`mask_image`、`height`、`width` 参数。
- 当前配置的 VLM endpoint 和 `qwen3.6` 是否支持 OpenAI-compatible `image_url` 输入。
- 当前 VLM provider 是否接受 `response_format={"type": "json_object"}`。
- panorama 质量、style consistency、artwork 等困难输入下的边界 artifact，未通过生成结果确认。
- soft inpainting mask 在真实 backend 中是否足以减少边缘 artifact。
- overlap blending 是否足以减少 seams，需要生成结果视觉确认。
- per-view refinement 是否视觉上等价于论文描述的 partial denoising。
- 论文没有充分公开 sampler、guidance scale、seed handling、blend kernel、mask post-processing 等低层参数，因此无法证明完全低层对齐。
- 当前依赖版本没有 requirements 文件固定，复现性依赖本机已有 `prmcam` conda environment。

## 当前复现状态

| 论文部分 | 状态 |
|---|---|
| Section 3.1 panorama generation 的 high-level anchored method | 基本实现 |
| Section 3.1 exact model 与 paper defaults | 部分对齐 |
| Section 3.1 prompt ablation modes | 部分实现 |
| Section 3.1 quantitative evaluation | 未实现 |
| Section 3.2 point-cloud-conditioned inpainting | 未实现 |
| Section 3.3 3D Gaussian Splatting reconstruction | 未实现 |
| 完整论文目标：2 m cube 内可导航的 3D world | 未实现 |

## 建议后续步骤

1. 新增 `pipeline-paper.toml`，使用论文尺度默认值：W x H 约 4096 x 2048 panorama、1024 view size、85-degree middle FoV、120-degree vertical FoV。
2. 增加可选 Dust3R FoV estimation，保留当前 manual `input_fov_x` 作为 fallback。
3. 增加 geometry-only visual debug checks，验证 projection round trip 和 yaw/pitch orientation。
4. 对 prompt modes 增加一键批量三路对比脚本，若需要正式实验表格。
5. 如果目标继续限定为 panorama-only，应在文档中明确 Section 3.2 和 3.3 out of scope。
6. 如果目标升级为 full-paper reproduction，应单独规划 MoGE/Metric3D、point-cloud-conditioned inpainting 和 Splatfacto。
