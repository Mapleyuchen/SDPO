# 角色设定
你现在是一位顶级的深度学习架构师和 PyTorch/RLHF 专家。你需要协助我（主导强化学习的 Member A）改造一个基础的开源 SDPO（Self-Distillation Policy Optimization）代码库，将其升级为我们团队为了投递 NeurIPS 2026 而设计的 **IsoGraph (Active-Symbolic SDPO)** 多模态强化学习框架。

# 项目背景与理论架构 (IsoGraph)
我们的任务是修复多模态大模型（MLLM）在解析复杂古籍排版时的“空间拓扑幻觉”。
当前的 SDPO 开源库是用于纯文本任务的一步生成式强化学习。而我们的框架包含一个 **VE-MDP (Visual-Evidence Markov Decision Process) 交互环境**和 **DGR (Diagnostic Graph Report) 富文本诊断反馈**。

我们需要对当前代码库进行“外科手术式”的三大改造。请你阅读以下核心改造需求，并在后续的交互中，一步步指导我修改对应的 Python 文件。

## 改造任务 1：植入 Action Interceptor (动作拦截器与环境交互)
传统的 `generate()` 是一口气输出到底的。我们需要把它改造成一个能与外部环境交互的流式循环。
* **需求：** 当 MLLM 的自回归生成遇到特定特殊 token（如 `<zoom>` 或 `<call_svm>`）时，必须**挂起 (Suspend) 生成**。
* **动作处理：**
    * 解析出 `<zoom> [x1, y1, x2, y2] </zoom>` 或 `<call_svm>`。
    * 调用一个占位符函数 `env.step(action)` 获取环境反馈（目前我会提供一个 `DummyEnvironment` 类来返回固定的字符串，如 "[System: SVM extracted text X]"）。
* **恢复生成：** 将环境返回的字符串拼接到 Context 窗口中，再次调用 MLLM 继续生成，直到输出最终答案或遇到 EOS。
* **目标：** 修改 Trajectory Rollout 逻辑，收集由 (状态, 动作, 环境反馈) 交替组成的完整轨迹 $y_{<t}$。

## 改造任务 2：Dual-Role Forward Pass (双角色前向传播与梯度阻断)
这是 SDPO 算法的心脏。我们需要在 Trainer 的 Loss 计算逻辑中进行修改。
* **Student Pass ($\pi_\theta$)：** 接收 Rollout 阶段的原始 prompt 和生成的轨迹，进行一次前向传播，计算并收集 action token 的 `student_logprobs`。**（这里必须保留梯度，用于后续 PPO 更新）**
* **Self-Teacher Pass ($\pi_{\theta'}$)：** 接收同样的轨迹，但在 Prompt 尾部**强行拼接**一段来自环境的富文本诊断报告 $f_{DGR}$ (Diagnostic Graph Report)。
    * **极其关键：** 这个前向传播必须被完全阻断梯度！(必须使用 `with torch.no_grad():` 或将 `teacher_logprobs` 进行 `.detach()`)。
    * 这里最好支持使用 EMA (指数移动平均) 更新的 Teacher 模型权重，如果代码库里没有 EMA，请先用当前的 Student 模型配合 `no_grad` 充当临时 Teacher。

## 改造任务 3：Token-Level Advantage 与 KL Penalty 截断目标
请检查并修改原仓库的 Loss 函数，确保严格实现了我们的数学公式：
1.  **细粒度优势估计：** 计算 Token 级别的概率差异：$A_t = \log \pi_{\theta'}(a_t|s_t, y_{<t}, f_{DGR}) - \log \pi_\theta(a_t|s_t, y_{<t})$。
2.  **优势归一化：** 必须对 $A_t$ 进行均值方差归一化，防止 Loss 爆炸。
3.  **Token 级 KL 惩罚：** 加载一个冻结的 Reference Model $\pi_{ref}$，计算当前 Student 与它的 Token 级 KL 散度。
4.  **最终的 Clip Loss：** 结合 PPO 的 Importance Sampling Ratio ($\rho_t$) 和 Clip 机制，将 KL 惩罚融入到最终的 Actor Loss 中计算梯度。

# 对 Cursor 的工程约束与要求：
1.  **显存敏感：** 我们处理的是 MLLM，极易 OOM。在写张量操作和 Forward pass 时，时刻注意释放不必要的计算图，熟练使用 `bfloat16`、`FlashAttention` 和 `torch.cuda.empty_cache()` 的理念。
2.  **张量形状对齐：** 在计算 `logprobs` 时，务必帮我严格检查 `input_ids` 和 `logits` 错位（shift）导致的 Shape Mismatch 问题（这是强化学习最容易踩的坑）。
3.  **第一步行动：** 请先扫描当前工作区的代码，找到 Rollout 生成（轨迹收集）和 Trainer/Loss 计算的核心文件，然后告诉我你打算从哪里开始下刀修改。我们先写一个 `DummyEnvironment` 把流程跑通！