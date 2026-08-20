import numpy as np
from cmaes import CMA

# ====================== 1.模拟环境（替代真实embedding、LLM rollout） ======================
# 工具库 5个工具
tools = ["T0_weather", "T1_stock", "T2_websearch", "T3_calc", "T4_sendmail"]
n_tool = len(tools)
# 模拟预计算好的语义相关性 I[v]
I = np.array([0.1, 0.92, 0.88, 0.3, 0.75])
# 模拟协同矩阵 S[v][S_set]：简化：S[v][s]表示v与集合s的协同，这里简化为预定义协同分数
synergy_raw = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.9, 0.2, 0.85],
    [0.0, 0.9, 0.0, 0.2, 0.8],
    [0.0, 0.2, 0.2, 0.0, 0.1],
    [0.0, 0.85, 0.8, 0.1, 0.0],
])
# 工具代价 C[v]
C = np.array([0.1, 0.3, 0.4, 0.1, 0.35])

max_select_k = 3  # 最多选3个工具

def proxy_delta(v_idx: int, selected: set, alpha: np.ndarray):
    r"""轻量级代理收益函数 \tilde{\Delta}，CPU运算，无LLM"""
    alpha_I, alpha_S, alpha_C = alpha
    i_score = I[v_idx]
    # 简化协同：与已选集合平均协同
    if len(selected) == 0:
        s_score = 0.0
    else:
        s_score = float(np.mean([synergy_raw[v_idx][s] for s in selected]))
    c_score = C[v_idx]
    delta = alpha_I * i_score + alpha_S * s_score - alpha_C * c_score
    return delta

def nested_greedy(alpha: np.ndarray):
    """内层嵌套贪心NGA，给定权重alpha，输出选出的工具下标集合"""
    selected = set()
    all_idx = set(range(n_tool))
    for _ in range(max_select_k):
        candidates = all_idx - selected
        best_v = None
        best_score = -np.inf
        for v in candidates:
            sc = proxy_delta(v, selected, alpha)
            if sc > best_score:
                best_score = sc
                best_v = v
        selected.add(best_v)
    return sorted(list(selected))

def simulate_rollout_reward(selected_idx):
    """仿真LLM rollout返回的奖励；真实项目这里替换成调用Agent rollout
    增加少量随机噪声，模拟真实LLM的随机性，避免开局直接满分
    """
    # 最优组合：T1_stock,T2_websearch,T4_sendmail → 奖励最高
    good = {1,2,4}
    hit = len(good & set(selected_idx))
    reward = hit / 3.0
    # 模拟毒丸惩罚：同时选T0+T3会扣分
    if 0 in selected_idx and 3 in selected_idx:
        reward *= 0.4
    # 增加噪声，模拟真实rollout的随机性
    noise = np.random.normal(0, 0.04)
    reward = np.clip(reward + noise, 0.0, 1.0)
    return reward

# ====================== 2. ENGA 演化主循环（离线阶段） ======================
def enga_evolution():
    # 优化变量 alpha = [alpha_I, alpha_S, alpha_C] 三维
    dim = 3
    # CMA‑ES初始化
    optimizer = CMA(
        mean=np.array([0.5, 0.5, 0.5]),
        sigma=0.3,
        bounds=np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0]]),
        seed=42
    )
    max_generations = 12
    best_reward = -1.0
    best_alpha = None
    best_subset = None

    for gen in range(max_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            alpha = optimizer.ask()
            # 内层贪心，CPU，不调用LLM
            subset = nested_greedy(alpha)
            # 种群级评估：仿真rollout奖励（真实工程此处调用Agent rollout）
            r = simulate_rollout_reward(subset)
            solutions.append((alpha, -r))  # cmaes做最小化，所以取负

            if r > best_reward:
                best_reward = r
                best_alpha = alpha.copy()
                best_subset = subset.copy()
        optimizer.tell(solutions)
        subset_names = [tools[i] for i in best_subset]
        print(f"Gen {gen:2d} | best_reward={best_reward:.3f} | best subset: {subset_names} | alpha={np.round(best_alpha,3)}")

    return best_alpha, best_subset, best_reward

if __name__ == "__main__":
    best_alpha, best_subset, best_r = enga_evolution()
    print("\n===== 演化结束，离线保存最优参数 =====")
    print(f"最优权重alpha: {np.round(best_alpha,3)}")
    print(f"最优工具子集: {[tools[i] for i in best_subset]}")
    print(f"仿真任务奖励: {best_r:.3f}")

    # ========== 线上推理流程（无rollout，仅CPU） ==========
    online_subset = nested_greedy(best_alpha)
    print("\n【线上推理】输入query，使用保存好的alpha，输出筛选后的工具子集：")
    print([tools[i] for i in online_subset])
    print("-> 将以上工具子集送入LLM Plan‑and‑Execute，执行function‑calling")


  
  cost:token+latency
  归一化：(real_value - min_value) / (max_value - min_value)