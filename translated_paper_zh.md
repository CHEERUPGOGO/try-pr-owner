# NESTED GREEDY SEARCH FOR TOOL SELECTION
**用于工具选择的嵌套贪心搜索**

*Under review as a conference paper at ICLR 2026*  
*匿名作者 (Anonymous authors)*  
*双盲评审中 (Paper under double-blind review)*  

---

## 摘要 (ABSTRACT)

LLM 的工具选择本质上是一个复杂的组合优化问题。当候选工具库规模庞大时，穷举评估所有工具组合是不切实际的（NP-hard）。虽然标准贪心算法效率较高，但它们往往无法兼顾工具之间复杂的相互作用。嵌套贪心算法（Nested Greedy Algorithm, NGA）通过在两层循环中递进选择工具，能够在理论上突破严格亚模性（Submodularity）的限制。然而，在实际 Agent 系统中直接应用 NGA 会面临三大严峻瓶颈：昂贵的大模型推理引发的计算爆炸、静态启发式目标带来的近视偏差，以及缺乏回溯机制导致的局部最优陷阱。为了解决这些痛点，本文提出了演化嵌套贪心算法（Evolutionary Nested Greedy Algorithm, ENGA）。ENGA 将高昂的大模型推理与内层工具搜索解耦，引入受超参数 $\alpha$ 控制的轻量级代理启发式函数 $\tilde{\Delta}$，并在外层利用演化算法（EA）对参数空间进行全局探索与自适应优化。此外，我们还将 ENGA 扩展至动态与顺序决策场景（Dynamic ENGA / DIN-GA），使 Agent 能够根据实时环境状态（如 DOM 树、API 交互反馈）动态调整工具选择策略。

---

## 1 预备知识 (PRELIMINARY)

### 1.1 形式化定义 (FORMULATION)

大语言模型（LLM）的工具选择本质上是一个复杂的组合优化问题。当候选工具库 $V$ 规模庞大时，穷举评估所有可能的工具组合是不切实际的（NP-hard 问题）。虽然标准贪心算法计算效率高，但它们往往无法处理工具之间复杂的协同与冲突关系。

### 1.2 NGA 的局限性 (LIMITATION OF NGA)

算法 1 展示了传统的嵌套贪心算法（Nested Greedy Algorithm, NGA）。

```text
Algorithm 1 Nested Greedy Algorithm (NGA) / 算法 1 嵌套贪心算法 (NGA)
Input: Ground set V ; budget b; solution-set size k / 输入：基础工具全集 V；预算 b；解集大小 k
Output: Solution set S / 输出：解集 S
1: S ← ∅
2: for j = 1 ... k do
3:     X_j ← ∅
4:     for t = 1 ... b do
5:         v* ← arg max_{v ∈ V \ X_j} Δ_{p_S}(v | X_j)
6:         X_j ← X_j ∪ {v*}
7:     end for
8:     S ← S ∪ X_j
9: end for
return S
```

在 LLM 工具选择和复杂推理任务中，传统嵌套贪心算法（NGA）虽然在理论上突破了严格亚模性的限制，但在实际工程部署中展现出显著的劣势。演化嵌套贪心算法（ENGA）正是为了解决这些痛点而提出的。

#### NGA 的局限性 (Limitations of NGA)

1. **计算开销爆炸（计算不可可行性）**：NGA 的内层循环需要频繁计算边际收益 $\Delta$。在 LLM 工具选择中直接应用 NGA 意味着每评估一个候选工具，都需要执行一次大模型的前向推理或完整 Rollout。对于庞大的工具库 $V$，其高达 $O(|V|^2)$ 的 LLM API 调用成本在时间和经济开销上都是不可接受的。
2. **近视启发式偏差**：NGA 依赖预先定义的静态目标函数 $h(S)$。在复杂的 Agent 场景中，工具的“质量”是多维度的（包含相关性、互斥性、延迟和 Token 成本）。人类专家很难任意指定一个完美的、静态的权重比例。如果初始目标函数定义不当，NGA 将坚定地收敛到次优解。
3. **易陷于局部最优**：尽管 NGA 比标准贪心算法具有更强的“前瞻”能力，但它本质上仍遵循确定性的贪心路径。一旦在外层循环早期做出了错误的选择（例如选择了一个看似相关但后续无法与其他工具协同的“毒丸”工具），NGA 缺乏回溯机制，无法逃离局部最优陷阱。

#### ENGA 的优势 (Advantages of ENGA)

1. **计算复杂度大幅降低（推理与搜索解耦）**：ENGA 将昂贵的大模型推理（适应度评估）与候选集的内层循环构建彻底解耦。内层循环采用受超参数 $\alpha$ 控制的轻量级代理函数 $\tilde{\Delta}$ 执行超快速计算（仅涉及向量点积和标量运算）。大模型调用的频率已从 NGA 中的“工具级探测”降低到 ENGA 的“种群级评估”，显著降低了系统延迟。
2. **自适应多目标调节**：ENGA 将平衡不同维度收益的挑战委托给演化算法。通过将语义相关性 ($I$)、工具协同性 ($S$) 和系统执行成本 ($C$) 抽象为权重向量 ($\\alpha$)，系统能为特定任务动态演化出最佳权重分布，无需手动调节超参数。
3. **基于种群动态的全局探索**：演化算法（如 CMA-ES 或遗传算法）天生具备全局优化能力。通过维持参数的多样化种群，并借助交叉与变异引入随机扰动，ENGA 能够探索更广阔的解空间，有效避免了确定性贪心算法在复杂非凸空间中容易出现的过早收敛问题。
4. **适配黑盒 LLM**：LLM 输出生成、幻觉概率及实际任务成功率通常是不可导的。ENGA 绕过了基于梯度的优化；它仅需要大模型提供最终的标量奖励（适应度）来驱动工具组合逻辑的迭代优化，非常适合当前 Agent 系统的工程范式。

---

### 1.3 演化嵌套贪心算法 (EVOLUTIONARY NGA - ENGA)

通过对 NGA 内层循环中计算高昂的边际收益 $\Delta$ 进行多目标参数化（引入 $\alpha_1, \alpha_2$），并使用演化算法（EA）求解，成功构建了一种混合优化架构。在 LLM 工具使用的复杂环境中，该设计巧妙地绕过了大语言模型黑盒且不可导的特性。

全新 ENGA 的核心创新：

1. **边际收益 $\Delta$ 的参数化拆解**  
   原始 NGA 需要在内层循环中实际评估 $\Delta_{p_S}(v \mid X_j)$；在 Agent 场景中，这意味着极其耗时的完整推理过程。通过引入权重参数 $\alpha$，“真实边际收益”可以通过降维近似为一个低成本的代理启发式函数：
   $$\tilde{\Delta}(v \mid X_j) = \alpha_1 \cdot \phi_{utility}(v) + \alpha_2 \cdot \phi_{synergy}(v, X_j) - \alpha_3 \cdot \phi_{cost}(v)$$
   - $\phi_{utility}(v)$：工具 $v$ 的固有实用性（例如基于历史调用成功率的先验得分）。
   - $\phi_{synergy}(v, X_j)$：工具 $v$ 与当前已选工具集 $X_j$ 之间的协同性或多样性（例如通过工具 Embedding 的正交性计算）。
   - $\phi_{cost}(v)$：调用该工具带来的 Token 消耗或延迟成本。  
   在该框架下，NGA 内层循环中的贪心选择步骤不再依赖模型 Rollout，而是转变为快速的标量计算。

2. **使用演化算法（EA）“求解”该系统的两条路径**

   - **路径 A：演化超参数（演化启发式规则）**  
     在此方案中，演化算法中的“个体”是一个权重向量 $\alpha = [\alpha_1, \alpha_2, \dots]$。
     1. **初始化种群**：随机生成多组 $\alpha$。
     2. **快速评估（NGA 作为解码器）**：对于种群中的每组 $\alpha$，将其带入内层函数 $\tilde{\Delta}$，快速执行 NGA 算法，并输出最终的工具组合 $S^{NGA}_{\alpha}$。
     3. **计算适应度**：将工具集 $S^{NGA}_{\alpha}$ 提交给 LLM，并在验证集上执行；所得的实际任务成功率（或综合奖励）作为该 $\alpha$ 的适应度。
     4. **变异与交叉**：舍弃低分 $\alpha$ 值，通过交叉与变异生成新权重，直到收敛。  
     *优势*：该方法利用了 CMA-ES（协方差矩阵自适应演化策略）等算法在连续空间中的强大优化能力，同时大幅减少了 LLM 调用次数。

   - **路径 B：多目标演化优化 (MOEA)**  
     如果您不想强行对不同位面的目标（如准确率与延迟）进行加权求和，可以将 $\alpha_1$ 和 $\alpha_2$ 视为两个独立的优化目标（例如 $f_1 = \text{准确率}, f_2 = \text{工具集冗余度}$）。通过使用如 NSGA-II 等多目标演化算法，您可以直接导出 Pareto 最优工具组合集。  
     *优势*：在部署期间，您可以根据当前系统负载沿 Pareto 前沿动态选择不同的工具组合（例如在非高峰期选择准确率极高但工具较多、耗时较长的组合，在高峰期切换为低延迟组合）。

#### 为什么这种方法高度可行？
- **避开了梯度缺失问题**：LLM 工具选择本质上是一个离散组合优化问题（0/1 背包问题的变体）；基于梯度的深度强化学习方法（如 PPO）在此类稀疏奖励与爆炸性动作空间中往往难以收敛。相比之下，演化算法（EA）是纯粹的免梯度全局优化器。
- **突破局部最优**：传统贪心算法极易在初始步骤犯错。通过使用 EA 持续扰动 $\alpha$——甚至直接对工具集 $S$ 进行交叉与重组——算法获得了摆脱局部最优所需的“逃逸能力”。

---

### 1.4 三种不同的应用场景 (SCENARIOS)

在 LLM 和 Agent 中，工具选择/函数调用是使模型突破纯文本生成局限、与外部系统进行实质性交互的核心机制。主要包含以下三大典型场景：

1. **信息检索与知识增强 (RAG)**
   - *单步 / 静态选择 / 单次决策*
   - 核心思想：$\text{查询} \to (\text{工具选择}) \to \text{工具} \to (\text{执行}) \to \text{响应}$
   - 系统仅需根据用户当前查询从工具库中识别出 Top-$k$ 最相关的工具。工具之间不存在因果链，也不存在动态反馈（例如执行工具 A 后页面或系统状态发生改变，进而解锁工具 B）。

2. **数据分析与执行**
   - *混合 / 半动态 / 取决于任务难度*
   - **静态模式（简单查询）**：用户询问“上个月哪种产品销量最高？”LLM 选择 SQL 工具，生成 SQL 语句并检索结果。这是典型的静态单步流程。
   - **动态模式（迭代分析 / 代码解释器）**：LLM 选择 Python 工具执行数据处理脚本。代码执行失败（生成新状态 $s_t$）。LLM 捕获错误信息并动态评估下一步：是切换到绘图工具、重写代码，还是使用数据清洗 API？

3. **多步 Agent 工作流**
   - *多步 / 顺序决策*
   - 核心思想：$s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} \dots s_n$
   - 例如在 Web 导航（如 WebArena Zhou et al. (2024)）、复杂跨服务工作流以及大模型 Agent 规划场景中，任务被显式建模为部分可观察马尔可夫决策过程（POMDP）。
   - **依赖状态**：上一步中工具 $a_t$ 的选择与执行（例如在预订网站上点击“搜索航班”）将环境状态转移至 $s_{t+1}$（例如到达航班列表页面）。下一步可用或恰当的工具集完全取决于这一新状态 $s_{t+1}$。
   - **顺序依赖**：工具的选择涉及严格的前置条件和后置条件，无法仅凭初始查询预先完全确定。

---

## 表 1：三种场景的总结与对比 (Table 1: Summarization of 3 Scenarios)

| 维度 (Dimension) | 静态工具选择 (Static Tool Selection) | 动态 / 顺序工具选择 (Dynamic / Sequential Tool Selection) |
| :--- | :--- | :--- |
| **主要映射场景**<br>(Primary Mapping Scenarios) | 场景 1 及简单场景 2 | 场景 3 及复杂场景 2（Web 导航等） |
| **核心决策公式**<br>(Core Decision Formula) | $v^* = \arg\max_v \text{Relevance}(v, q)$ | $v^*_t = \arg\max_v \tilde{\Delta}(v \mid X_{<t}, s_t)$ |
| **工具交互关系**<br>(Tool Interaction) | 独立或并行选择（无因果关系） | 强依赖性、顺序性、协同或冲突 |
| **环境反馈机制**<br>(Environmental Feedback) | 无反馈或仅返回单次结果 | 实时环境状态变化（DOM 树更新、API 状态变更） |
| **优化重点**<br>(Optimization Focus) | 语义相关性、召回率 | 状态对齐、时间连贯性、执行风险管理 |

---

### 1.5 总结 (SUMMARIZATION)

总结了静态与动态工具选择之间的内在联系与区别。

---

## 2 方法 (METHODOLOGY)

### 2.1 NGA
介绍了嵌套贪心算法的基本形式与数学基础。

### 2.2 静态 ENGA (STATIC ENGA)

#### 目标函数
$$\tilde{\Delta}(v \mid X_j) = \alpha_1 \cdot I_{rel}(v, q) + \alpha_2 \cdot S_{syn}(v, X_j) - \alpha_3 \cdot C_{run}(v) \quad \text{--- 公式 (1)}$$

1. **$\alpha_1$: 语义相关性 (Semantic Relevance) – 侧重工具表征**  
   衡量候选工具 $v$ 与当前用户查询 $q$ 或当前 RAG/QA 上下文之间的直接匹配程度。
   - **物理意义**：任务意图的初始对齐（Intent Alignment）。
   - **计算方法**：可采用稠密向量检索技术。通过对工具描述（Tool Description/Schema）与当前用户查询进行 Embedding，将 $I_{rel}$ 定义为两者在高维空间中的余弦相似度。

2. **$\alpha_2$: 协同性与信息增益 (Synergy & Information Gain) – 侧重结构化表征**  
   在 RAG/QA Agent 工作流中，工具常展现出强补完性。如果已经选择了“基于内容的检索工具”，再选择另一个高度相似的工具将导致边际效用显著递减。
   - **物理意义**：工具组合的非冗余性或互信息。
   - **计算方法**：
     - **正交性惩罚**：计算候选工具 $v$ 的 Embedding 与当前已选集合 $X_j$ 中所有工具 Embedding 的相似度，取最大值作为惩罚项（反之可定义为多样性增益）。
     - **基于 Schema 的互信息**：评估工具 $v$ 的输出数据结构能否作为 $X_j$ 内某工具的输入。特别是在通过 Model Context Protocol (MCP) 串联工具时，如果工具 A 的输出精准契合工具 B 的输入参数规范，这种拓扑连通性将产生极高的 $S_{syn}$ 得分。

3. **$\alpha_3$: 运行惩罚 (Runtime Penalty) – 侧重反馈指标**  
   在生产部署中，大语言模型（LLM）的上下文窗口和响应延迟构成了硬约束。
   - **物理意义**：与系统执行相关的“摩擦成本”。
   - **计算方法**：
     - **Token 消耗**：工具 Schema 描述的长度以及预期返回结果的平均 Token 数。
     - **时间延迟**：对特定 API 或工具历史调用的平均延迟。
     - **失败率**：从历史日志计算出的该工具触发 LLM 幻觉或执行错误的概率。  
   作为负约束，防止 EA 演化出过于臃肿的工具链（包含十几工具导致严重的系统超时或超出 Token 限制）。

在三维框架下，演化算法（EA）优化向量 $\alpha = [\alpha_1, \alpha_2, \alpha_3]$。不同 RAG/QA 场景下 $\alpha$ 的最佳分布有所不同。例如：
- 对于简单的重复购买查询，EA 可能会演化出具有极高 $\alpha_3$ 的权重组合（优先考虑低延迟）。
- 对于高度模糊、探索性的 RAG/QA 任务（如“帮我规划一份完整的夏季露营装备清单”），EA 会演化出具有极高 $\alpha_2$ 的权重组合（鼓励工具间的高协同性与多样性）。

#### 核心思想 (Key Idea)
外层采用演化算法（EA）演化权重向量 $\alpha$，内层利用嵌套贪心算法（NGA）结合代理启发式函数快速构建工具子集，最后由 LLM 进行实际的适应度评估。

```text
Algorithm 2 Evolutionary Nested Greedy Algorithm (ENGA) for General Tool Selection
算法 2 通用工具选择的演化嵌套贪心算法 (ENGA)
Input: Candidate tool library V; Outer budget k; Inner budget b; User query q; Population size P; Max generations G
输入：候选工具库 V；外层预算 k；内层预算 b；用户查询 q；种群大小 P；最大代数 G
Output: Optimal tool subset S* / 输出：最佳工具子集 S*
1: Initialize population P ← {α^(1), α^(2), ..., α^(P)}, where α = [α1, α2, α3]
2: Global best fitness F* ← -∞
3: Global best tool subset S* ← ∅
4: for generation g = 1 ... G do
5:     for each individual α ∈ P do
6:         // Phase 1: Fast NGA Decoding using Proxy Heuristic / 阶段 1：使用代理启发式的快速 NGA 解码
7:         S_α ← ∅
8:         for j = 1 ... k do
9:             X_j ← ∅  ▷ Initialize inner tool bundle / 初始化内层工具包
10:            for t = 1 ... b do
11:                // Calculate proxy marginal gain ~Δ / 计算代理边际收益 ~Δ
12:                v* ← arg max_{v ∈ V \ X_j} [α1 · I_rel(v, q) + α2 · S_syn(v, X_j) - α3 · C_run(v)]
13:                X_j ← X_j ∪ {v*}
14:            end for
15:            S_α ← S_α ∪ X_j
16:        end for
17:        // Phase 2: Fitness Evaluation via LLM Execution / 阶段 2：通过 LLM 执行进行适应度评估
18:        F(α) ← EvaluateFitness(S_α, q)  ▷ Execute constructed toolset and get reward / 执行构建的工具集并获取奖励
19:        if F(α) > F* then
20:            F* ← F(α)
21:            S* ← S_α
22:        end if
23:    end for
24:    // Phase 3: Evolutionary Operations (e.g., CMA-ES or Genetic Algorithm) / 阶段 3：演化操作
25:    P_parents ← Selection(P, F)  ▷ Select elites based on fitness / 根据适应度选择精英
26:    P ← CrossoverAndMutation(P_parents)  ▷ Generate new weights for next generation / 为下一代生成新权重
27: end for
return S*
```

#### 算法 2 步骤详解 (Explanation)
1. **输入参数解耦**：算法抽象掉了具体的业务逻辑；输入仅由工具库 $V$、预算约束 ($k, b$) 以及通用查询 $q$ 组成。
2. **阶段 1（快速解码）**：该步骤纯粹涉及轻量级数学运算（如向量点积和相似度检索）。使用当前代的权重 $\alpha$，快速构建包含 $k \times b$ 个工具的候选集 $S_\alpha$。这避免了在内层循环中调用大语言模型（LLM）。
3. **阶段 3（真实评估）**：对完整构建的工具集 $S_\alpha$ 执行单次“真实”的 `EvaluateFitness` 评估（例如将其喂给 LLM 执行并计算任务成功率、综合奖励等指标），并将结果作为当前超参数 $\alpha$ 的适应度得分。
4. **阶段 4（种群演化）**：通过选择、交叉和变异等遗传算法操作，驱动权重 $\alpha$ 向能够产生更高适应度工具集的方向演化。

---

### 2.3 动态 ENGA (DYNAMIC ENGA)

#### 目标函数
$$\tilde{\Delta}(v \mid X_j, s_t) = \alpha_1 \cdot I_{rel}(v, q, s_t) + \alpha_2 \cdot S_{syn}(v, X_j) - \alpha_3 \cdot C_{run}(v) \quad \text{--- 公式 (2)}$$

1. **$\alpha_1$: 状态-目标对齐 (State-Goal Alignment)**  
   在 Web 环境中，工具的相关性不仅取决于全局目标 $q$（例如“明天帮我预订去纽约的早班机票”），更关键的是取决于当前页面状态 $s_t$（例如当前处于“乘客信息”页面）。
   - **物理意义**：在给定当前 DOM 状态下，候选动作 $v$ 在推进全局任务 $q$ 方面的局部实用性。
   - **计算方法**：提取与候选动作 $v$ 关联的 UI 元素的文本和属性（例如 `<button aria-label="Search Flights">`）。使用轻量级编码器计算该元素的 Embedding，并计算其与当前子任务意图（由目标 $q$ 和当前状态 $s_t$ 导出）的相似度。
   - **场景映射**：如果当前状态 $s_t$ 是首页且目标是订机票，则动作 $v = \text{type}[\text{Input Destination}, \text{"JFK"}]$ 将获得极高得分，而 $v = \text{click}[\text{FAQ Link}]$ 将获得极低得分。

2. **$\alpha_2$: 顺序依赖与连贯性 (Sequential Dependency & Coherence)**  
   WebArena 中的动作受严格的因果关系支配——即“前置条件”和“后置条件”。例如，在选择日期之前不能点击“搜索”，在同意条款之前不能点击“支付”。
   - **物理意义**：候选工具 $v$ 与已执行动作序列 ($X_j$) 之间的逻辑连贯性；具体而言，$X_j$ 产生的状态是否满足 $v$ 所需的前置条件。
   - **计算方法**：
     - **基于 Schema 的前/后置条件匹配**：例如，动作 `click[Search]` 的前置条件通常要求特定输入框已被填充。通过定义或学习转移矩阵，可以计算给定历史轨迹 $X_j$ 下动作 $v$ 的条件转移概率 $P(v \mid X_j)$。
     - **DOM Focus 追踪**：评估被 $v$ 作用的 UI 元素与 $X_j$ 中刚交互过的元素是否在物理空间或逻辑层级上保持合理连续性（例如在填写“姓”输入框后紧接着填写“名”输入框）。
   - **场景映射**：如果 $X_j$ 包含 `click[Departure Date Input]`，则后续候选动作 $v = \text{click}[\text{Calendar Day 24}]$ 将获得显著的协同奖励。

3. **$\alpha_3$: 执行风险与不可逆性 (Execution Risk & Irreversibility)**  
   在 Web 导航中，执行错误动作带来的成本是不对称的。向下滚动页面的成本极低（因为可以通过向上滚动撤销），而点击“提交订单”或退出当前预订流程的成本极高，可能导致整个 Episode 失败。此外，网页通常极其冗长，处理复杂的 DOM 树会消耗大量 Token。
   - **物理意义**：动作的执行成本（Token 消耗）以及引发状态崩溃的潜在风险。
   - **计算方法**：
     - **Token / 时间成本**：估计系统在执行动作 $v$ 后必须解析的下一个状态的 DOM 复杂度。
     - **不可逆性惩罚**：为高风险动作分配惩罚系数。例如，表单提交、支付和跨域导航被归类为高风险动作，而悬停和滚动为低风险。  
   对于高风险动作，算法在评估其 $\Delta$ 值时采用更加保守的策略（即它们需要获得来自 $\alpha_1$ 和 $\alpha_2$ 的强有力支持才能被选中）。
   - **场景映射**：对于 $v = \text{click}[\text{Confirm Payment}]$，$C_{risk}$ 的值非常高。除非 $X_j$ 已经完美包含了所有必要步骤（如座位选择和表单填写，从而产生极高的 $S_{seq}$），否则算法不会在探索早期贪婪地选择它。

---

```text
Algorithm 3 Dynamic Interactive Nested Greedy Algorithm (DIN-GA) in WebArena
算法 3 WebArena 中的动态交互式嵌套贪心算法 (DIN-GA)
Input: WebArena Environment E; Goal query q; Horizon budget T; Weight vector α = [α1, α2, α3]
输入：WebArena 环境 E；目标查询 q；时间步预算 T；权重向量 α = [α1, α2, α3]
Output: Execution trajectory XT and task success status / 输出：执行轨迹 XT 及任务成功状态
1: Initialize WebArena State s1 ← E.reset(q)
2: Initialize historical action trajectory X0 ← ∅
3: for step t = 1 ... T do
4:     Parse current DOM/Accessibility tree from st to extract valid action candidates V(st)
5:     if st is terminal or V(st) = ∅ then
6:         break  ▷ Episode finished or task deadlocked
7:     end if
8:     // Phase 1: State-Conditioned Marginal Gain Evaluation / 阶段 1：状态条件下的边际收益评估
9:     for each candidate action v ∈ V(st) do
10:        Compute State-Goal Alignment: I_rel(v, q, st)
11:        Compute Sequential Dependency Score: S_seq(v, X_<t)
12:        Compute Execution Risk & Cost: C_risk(v)
13:        // Calculate parameterized dynamic marginal gain ~Δ
14:        ~Δ(v | X_<t, st) ← α1 · I_rel(v, q, st) + α2 · S_seq(v, X_<t) - α3 · C_risk(v)
15:    end for
16:    // Phase 2: Action Selection / 阶段 2：动作选择
17:    v*_t ← arg max_{v ∈ V(st)} ~Δ(v | X_<t, st)
18:    // Phase 3: Real Environment Interaction & State Transition / 阶段 3：真实环境交互与状态转移
19:    Execute action v*_t in WebArena: (st+1, rt, done) ← E.step(v*_t)
20:    Update action trajectory: Xt ← X_<t ◦ (st, v*_t)
21:    if done then
22:        break  ▷ Goal achieved or failure triggered
23:    end if
24: end for
return Trajectory Xt and final environment task status
```

#### 算法 3 步骤详解 (Explanation)
- **Line 4 (`GetValidActions`)**：在网页导航中，可用工具并非静态不变。在每个时间步，算法通过解析当前页面（DOM 树）动态提取交互元素（如文本输入框、可点击按钮、下拉菜单），构建该时间步的动作池 $V(s_t)$。
- **Lines 9–13 ($\tilde{\Delta}$ 的计算)**：
  - $I_{rel}$：评估在当前页面状态 $s_t$ 下，动作 $v$ 是否契合全局预订目标 $q$（例如在航班列表页上，选择“按价格排序”或“选择早班机”会获得更高的对齐得分）。
  - $S_{seq}$：评估动作 $v$ 与前 $t-1$ 步历史轨迹 $X_{<t}$ 的连贯性（例如在输入乘客姓名后立即输入身份证号）。
  - $C_{risk}$：对不可逆或高成本动作（如“提交支付”或“取消订单”）实施惩罚，以防止盲目探索。
- **Line 18 (`Environment Execution`)**：在 WebArena 模拟器中执行选定的动作 $v^*_t$，并获取网页刷新后的新状态 $s_{t+1}$，从而建立动态闭环过程。

---

## 参考文献 (REFERENCES)

1. Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. **Webarena: A realistic web environment for building autonomous agents**, 2024. URL https://arxiv.org/abs/2307.13854.
