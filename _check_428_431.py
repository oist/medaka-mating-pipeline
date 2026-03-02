"""一次性脚本：复现 428,429,430,431 四帧的代价矩阵与 Hungarian 解。不修改项目代码。"""
import numpy as np
from scipy.optimize import linear_sum_assignment

# player2 id_assigned 原始坐标（从 CSV 提取）
# 427
p427 = {"a": (260.60797119140625, 198.4510498046875), "b": (210.27496337890625, 208.3384246826172), "c": (260.157958984375, 184.5168609619141)}
# 428
p428 = {"a": (262.3487548828125, 198.31333923339844), "b": (208.26644897460935, 206.42015075683597), "c": (260.106201171875, 186.4375457763672)}
# 429 三个检测（track 按 CSV 出现顺序：0448, 0640, 0832 → 排序后列顺序不变，这里按 a,c,b 的分配结果写坐标）
d429 = [(1080.274658203125, 150.42642211914062), (262.4105529785156, 198.1942901611328), (206.08767700195312, 206.12310791015625)]
# 430
d430 = [(1080.143310546875, 150.111328125), (264.1405334472656, 198.18051147460935), (204.17141723632807, 204.40013122558597)]
# 431
d431 = [(1078.247314453125, 152.47702026367188), (264.4176330566406, 198.3140411376953), (202.194580078125, 204.3002014160156)]

fish_ids = ["a", "b", "c"]

def compute_cost(prev_pos, curr_pos, prev_prev_pos=None, flow_weight=0.0):
    n, m = len(prev_pos), len(curr_pos)
    cost = np.zeros((n, m))
    for i in range(n):
        p_prev = np.array(prev_pos[i], dtype=float)
        p_prev_prev = np.array(prev_prev_pos[i], dtype=float) if prev_prev_pos else None
        for j in range(m):
            p_curr = np.array(curr_pos[j], dtype=float)
            dist = float(np.linalg.norm(p_prev - p_curr))
            flow_penalty = 0.0
            if flow_weight > 0 and p_prev_prev is not None:
                v_prev = p_prev - p_prev_prev
                v_new = p_curr - p_prev
                n_prev = np.linalg.norm(v_prev)
                n_new = np.linalg.norm(v_new)
                if n_prev > 1e-6 and n_new > 1e-6:
                    cos_angle = np.dot(v_prev, v_new) / (n_prev * n_new)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    flow_penalty = (1.0 - cos_angle) * flow_weight
            cost[i, j] = dist + flow_penalty
    return cost

def run_step(name, prev_dict, curr_list, prev_prev_dict=None, flow_weight=10.0):
    prev_pos = [prev_dict[f] for f in fish_ids]
    prev_prev_pos = [prev_prev_dict[f] for f in fish_ids] if prev_prev_dict else None
    c = compute_cost(prev_pos, curr_list, prev_prev_pos, flow_weight)
    row_ind, col_ind = linear_sum_assignment(c)
    total = c[row_ind, col_ind].sum()
    assignment = {col_ind[i]: fish_ids[row_ind[i]] for i in range(3)}
    order = [assignment[j] for j in range(3)]
    return c, total, order

print("========== 428 -> 429 ==========")
print("428 上一帧位置: a={:.1f},{:.1f}  b={:.1f},{:.1f}  c={:.1f},{:.1f}".format(*(p428["a"]+p428["b"]+p428["c"])))
print("429 本帧三个检测: det0={:.1f},{:.1f}  det1={:.1f},{:.1f}  det2={:.1f},{:.1f}".format(*(d429[0]+d429[1]+d429[2])))
C, tot, order = run_step("428->429", p428, d429, p427, flow_weight=10.0)
print("代价矩阵 (行=a,b,c 列=det0,det1,det2), flow_weight=10:")
print(C)
print("Hungarian 总代价: {:.2f}".format(tot))
print("分配结果 (det0, det1, det2 分别得到):", order)
C0, tot0, order0 = run_step("428->429 no flow", p428, d429, p427, flow_weight=0.0)
print("若 flow_weight=0，总代价: {:.2f}，分配:", tot0, order0)
print()

print("========== 429 -> 430 ==========")
# 429 的「上一帧」：按 428->429 的分配结果，a/b/c 在 429 的位置
p429_assigned = {"a": d429[0], "c": d429[1], "b": d429[2]}  # 实际 CSV 分配为 det0=a, det1=c, det2=b
C2, tot2, order2 = run_step("429->430", p429_assigned, d430, p428, flow_weight=10.0)
print("代价矩阵 (flow_weight=10):")
print(C2)
print("总代价: {:.2f}  分配:", tot2, order2)
print()

print("========== 430 -> 431 ==========")
p430_assigned = {order2[0]: d430[0], order2[1]: d430[1], order2[2]: d430[2]}
C3, tot3, order3 = run_step("430->431", p430_assigned, d431, p429_assigned, flow_weight=10.0)
print("代价矩阵 (flow_weight=10):")
print(C3)
print("总代价: {:.2f}  分配:", tot3, order3)
print()
print("四帧 428,429,430,431 最终每帧顺序: 428 按上一段 -> 429 {} -> 430 {} -> 431 {}".format(order, order2, order3))
