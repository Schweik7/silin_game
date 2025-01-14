from collections import deque
import heapq
import random
import copy
from typing import List, Optional, Dict, Tuple
from utils import Cube, PuzzleGenerator


# ------------------------------
#  通用的边匹配检查函数
# ------------------------------
def is_valid_placement(grid, r, c, cube, size):
    """
    检查在 grid[r][c] 放置 cube 是否满足相邻边匹配:
    - 上方 cube.down == 当前 cube.up
    - 左方 cube.right == 当前 cube.left
    - 已经放置的下方和右方同理
    """
    # 检查上方
    if r > 0 and grid[r - 1][c] is not None:
        if grid[r - 1][c].down != cube.up:
            return False

    # 检查左方
    if c > 0 and grid[r][c - 1] is not None:
        if grid[r][c - 1].right != cube.left:
            return False

    # 检查下方是否已放置
    if r < size - 1 and grid[r + 1][c] is not None:
        if grid[r + 1][c].up != cube.down:
            return False

    # 检查右方是否已放置
    if c < size - 1 and grid[r][c + 1] is not None:
        if grid[r][c + 1].left != cube.right:
            return False

    return True


# ==========================================================================
#                           一、A* 算法求解
# ==========================================================================


class AStarState:
    """
    A* 的状态：
      - grid:   当前已经放置的方块布局 (size x size)
      - used:   已经放置了哪些 cube（在原列表中的索引或者用 set 存储也行）
      - g_cost: 已经放置的数量(或已匹配边数)等，作为代价的一部分
      - h_cost: 启发式估计剩余放置的代价
    """

    __slots__ = ["grid", "used", "g_cost", "h_cost"]

    def __init__(self, grid, used, g_cost, h_cost):
        self.grid = grid
        self.used = used
        self.g_cost = g_cost
        self.h_cost = h_cost

    def __lt__(self, other):
        # Python 堆需要 < 比较，用 F = g+h 判断优先级
        return (self.g_cost + self.h_cost) < (other.g_cost + other.h_cost)


def heuristic(grid, size):
    """
    简易启发式：统计当前已经放置的匹配边数，与理论最大匹配数之差
    或者也可用“尚未放置的方块数量”作为 h。
    这里我们简单地用：h_cost = (未放置格子数)。
    """
    not_placed = 0
    for r in range(size):
        for c in range(size):
            if grid[r][c] is None:
                not_placed += 1
    return not_placed


def get_next_pos(r, c, size):
    """给定当前 (r, c)，返回下一个要放置的网格位置 (nr, nc)。"""
    nc = c + 1
    nr = r
    if nc >= size:
        nc = 0
        nr = r + 1
    return nr, nc


def reconstruct_solution(grid):
    """
    返回一个深拷贝，作为最终解。
    """
    import copy

    return copy.deepcopy(grid)


def astar_solve(cubes: List[Cube], size: int) -> Optional[List[List[Cube]]]:
    """
    使用 A* 算法求解拼图。
    :param cubes: 需要放置的方块列表
    :param size:  棋盘大小
    :return:      如果找到解，返回一个 size x size 的布局，否则 None
    """
    # 初始状态：grid 全部 None，used = set()
    start_grid = [[None for _ in range(size)] for _ in range(size)]
    start_used = set()
    start_h = heuristic(start_grid, size)

    start_state = AStarState(start_grid, start_used, g_cost=0, h_cost=start_h)

    # 优先队列（最小堆）
    open_list = []
    heapq.heappush(open_list, start_state)
    visited = set()

    while open_list:
        current_state = heapq.heappop(open_list)
        current_grid = current_state.grid
        current_used = current_state.used

        # 判断是否结束：如果 used 大小 == size*size，则说明全部放置完毕
        if len(current_used) == size * size:
            # 找到解
            return reconstruct_solution(current_grid)

        # 下一个要放置的 (r, c)
        # 简单从左到右、从上到下找第一个 None 的位置
        r, c = 0, 0
        found_pos = False
        for rr in range(size):
            for cc in range(size):
                if current_grid[rr][cc] is None:
                    r, c = rr, cc
                    found_pos = True
                    break
            if found_pos:
                break

        # 遍历所有可用的 cubes
        for idx, cube in enumerate(cubes):
            if idx not in current_used:
                # 检查放在 (r, c) 是否有效
                if is_valid_placement(current_grid, r, c, cube, size):
                    # 生成新的状态
                    new_grid = copy.deepcopy(current_grid)
                    new_grid[r][c] = cube
                    new_used = set(current_used)
                    new_used.add(idx)
                    g_cost = len(new_used)  # 已放置的方块数量
                    h_cost = heuristic(new_grid, size)
                    new_state = AStarState(new_grid, new_used, g_cost, h_cost)

                    # 去重判断
                    # 可以用状态的字符串表示来去重，这里仅示例
                    state_id = state_signature(new_state.grid, size, new_state.used)
                    if state_id not in visited:
                        visited.add(state_id)
                        heapq.heappush(open_list, new_state)

    # 如果队列空了还没有返回，说明无解
    return None


def state_signature(grid, size, used):
    """
    用于去重判断的签名：可以把 grid 中已经放置的 cube 的 id 拼起来。
    """
    # 最简单：把 used 做成一个不可变元组 + 保证顺序
    used_tuple = tuple(sorted(list(used)))
    return used_tuple


# ==========================================================================
#                           二、遗传算法求解
# ==========================================================================
def evaluate_arrangement(arr: List[Cube], size: int) -> int:
    """
    评价函数：给出一个方块排列 arr，其在 size x size 网格中
    从上到下、从左到右放置时，总的「匹配边」数量。
    分数越高越好（匹配越多）。
    """
    score = 0
    # 将 arr 映射到 grid
    grid = []
    idx = 0
    for r in range(size):
        row_cubes = []
        for c in range(size):
            row_cubes.append(arr[idx])
            idx += 1
        grid.append(row_cubes)

    # 统计相邻匹配
    for r in range(size):
        for c in range(size):
            if r < size - 1:
                # 下边比较
                if grid[r][c].down == grid[r + 1][c].up:
                    score += 1
            if c < size - 1:
                # 右边比较
                if grid[r][c].right == grid[r][c + 1].left:
                    score += 1
    return score


def crossover(p1: List[Cube], p2: List[Cube]) -> Tuple[List[Cube], List[Cube]]:
    """
    简单单点交叉：将两个排列在某个位置拆开并交换尾部。
    """
    size = len(p1)
    if size < 2:
        return p1, p2
    point = random.randint(1, size - 1)
    c1 = p1[:point] + [x for x in p2 if x not in p1[:point]]
    c2 = p2[:point] + [x for x in p1 if x not in p2[:point]]
    return c1, c2


def mutate(individual: List[Cube], mutation_rate: float):
    """
    简单变异：根据 mutation_rate 的概率，随机挑两个位置的方块进行交换。
    """
    for _ in range(len(individual)):
        if random.random() < mutation_rate:
            i = random.randint(0, len(individual) - 1)
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]


def genetic_solve(
    cubes: List[Cube],
    size: int,
    population_size=50,
    generations=500,
    mutation_rate=0.05,
) -> Optional[List[List[Cube]]]:
    """
    遗传算法求解：
    1. 个体：一个方块排列 (长度 = size*size)，表示按行填充到网格
    2. 评价函数：evaluate_arrangement
    3. 选择、交叉、变异
    :return: 如果找到完美解(所有边匹配)，则返回其 2D 网格；否则返回分数最高的
    """
    # 初始种群：随机打乱 cubes
    original = cubes.copy()
    population = []
    for _ in range(population_size):
        random.shuffle(original)
        population.append(original[:])  # copy

    best_arr = None
    best_score = -1

    perfect_score = (size - 1) * size + size * (size - 1)
    # 对于 NxN 网格，水平方向 (size * (size-1)) 条边，垂直方向一样，总匹配边数 = 2 * size * (size-1)

    for gen in range(generations):
        # 1. 评价所有个体
        scores = []
        for indiv in population:
            sc = evaluate_arrangement(indiv, size)
            scores.append((sc, indiv))

        # 更新最佳
        scores.sort(key=lambda x: x[0], reverse=True)
        if scores[0][0] > best_score:
            best_score = scores[0][0]
            best_arr = scores[0][1][:]
        # 如果达到完美匹配，则直接返回
        if best_score == perfect_score:
            print(f"在第 {gen} 代找到完美解，匹配总数={perfect_score}.")
            return arrangement_to_grid(best_arr, size)

        # 2. 选择（简易：截断选择）
        #    取前 1/2 个体作为精英
        half = population_size // 2
        new_population = []
        elites = scores[:half]

        # 3. 交叉
        while len(new_population) < population_size:
            p1 = random.choice(elites)[1]
            p2 = random.choice(elites)[1]
            c1, c2 = crossover(p1, p2)
            # 4. 变异
            mutate(c1, mutation_rate)
            mutate(c2, mutation_rate)
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = new_population

    print(f"遗传算法运行结束，最佳匹配数={best_score} / {perfect_score}.")
    if best_arr:
        return arrangement_to_grid(best_arr, size)
    else:
        return None


def arrangement_to_grid(arr: List[Cube], size: int) -> List[List[Cube]]:
    """
    将一维的方块排列映射回 2D 网格。
    """
    grid = []
    idx = 0
    for r in range(size):
        row_cubes = []
        for c in range(size):
            row_cubes.append(arr[idx])
            idx += 1
        grid.append(row_cubes)
    return grid


# ------------------------------
#  示例主函数
# ------------------------------
def demo_astar():
    """
    演示 A* 求解
    """
    generator = PuzzleGenerator(size=3, max_num=7)
    generator.print_initial_solution()
    # 打乱后的 cubes
    cubes = generator.get_random_cubes()
    print("===== A* 演示 =====")
    solution = astar_solve(cubes, size=3)
    if solution is None:
        print("A* 未找到解。")
    else:
        print("A* 找到的解：")
        print_grid(solution)


def demo_genetic():
    """
    演示 遗传算法 求解
    """
    generator = PuzzleGenerator(size=3, max_num=2)
    cubes = generator.get_random_cubes()
    print("===== 遗传算法 演示：3x3 拼图，数字范围 0~2 =====")
    solution = genetic_solve(
        cubes, size=3, population_size=50, generations=500, mutation_rate=0.05
    )
    if solution is None:
        print("遗传算法 未找到解（或匹配不完全）。")
    else:
        print("遗传算法 找到的解：")
        print_grid(solution)


def print_grid(grid: List[List[Cube]]):
    """
    打印最终解的网格布局
    """
    size = len(grid)
    for r in range(size):
        # up line
        up_line = " ".join(f"{grid[r][c].up:2d}" for c in range(size))
        print(up_line)

        mid_line = " ".join(
            f"{grid[r][c].left}*{grid[r][c].right}" for c in range(size)
        )
        print(mid_line)

        down_line = " ".join(f"{grid[r][c].down:2d}" for c in range(size))
        print(down_line)
        print()


if __name__ == "__main__":
    demo_astar()
    print("\n")
    # demo_genetic()
