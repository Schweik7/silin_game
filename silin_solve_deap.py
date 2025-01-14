import random
from typing import List
from deap import base, creator, tools, algorithms

from utils import Cube, PuzzleGenerator


def evaluate_arrangement(individual: List[Cube], size: int) -> int:
    """
    评价函数：将 individual（长度 = size*size 的 Cube 列表）映射为网格，
    计算相邻匹配的边数，返回匹配总数（越大越好）。
    
    满分（完美解）的匹配数 = 2 * size * (size - 1)。
      - 行方向匹配边：size * (size - 1)
      - 列方向匹配边：size * (size - 1)
    """
    # 将个体排列映射回网格
    grid = []
    idx = 0
    for r in range(size):
        row_cubes = []
        for c in range(size):
            row_cubes.append(individual[idx])
            idx += 1
        grid.append(row_cubes)

    score = 0
    # 检查行方向的相邻匹配
    for r in range(size):
        for c in range(size - 1):
            if grid[r][c].right == grid[r][c + 1].left:
                score += 1

    # 检查列方向的相邻匹配
    for r in range(size - 1):
        for c in range(size):
            if grid[r][c].down == grid[r + 1][c].up:
                score += 1

    return score


def arrangement_to_grid(arr: List[Cube], size: int) -> List[List[Cube]]:
    """
    将一维的个体（Cube 列表）映射回 2D 网格，用于最终结果打印/展示。
    """
    grid = []
    idx = 0
    for r in range(size):
        row = []
        for _ in range(size):
            row.append(arr[idx])
            idx += 1
        grid.append(row)
    return grid


def print_grid(grid: List[List[Cube]]):
    """
    打印最终解的网格布局。
    """
    size = len(grid)
    for r in range(size):
        # 第一行：up
        up_line = " ".join(f"{grid[r][c].up:2d}" for c in range(size))
        print(up_line)

        # 中间行：left*right
        mid_line = " ".join(f"{grid[r][c].left}*{grid[r][c].right}"
                            for c in range(size))
        print(mid_line)

        # 最后一行：down
        down_line = " ".join(f"{grid[r][c].down:2d}" for c in range(size))
        print(down_line)
        print()


def main():
    # --------------------------
    #  1. 生成拼图并获取所有方块
    # --------------------------
    size = 3
    max_num = 2  # 可以根据需要调节数字范围
    generator = PuzzleGenerator(size=size, max_num=max_num)
    cubes = generator.get_cubes()  # 原始有序列表
    random.shuffle(cubes)          # 打乱顺序

    # 需要确保个体长度等于 size*size
    assert len(cubes) == size * size, "PuzzleGenerator 生成的方块数量不等于 size*size!"

    # --------------------------
    #  2. 创建 DEAP 的遗传算法框架
    # --------------------------
    # 适应度：单目标最大化
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # 个体：直接用 list[Cube] 表示
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # 个体初始化：将原始的 cubes 做随机洗牌，得到一个 Individual
    def init_individual():
        # 这里复制一份 cubes 并打乱
        arr = cubes[:]
        random.shuffle(arr)
        return creator.Individual(arr)

    # 注册初始化函数：population(n) 会重复调用 init_individual
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 评价函数：计算相邻边匹配数
    def eval_ind(ind):
        return (evaluate_arrangement(ind, size),)

    toolbox.register("evaluate", eval_ind)
    # 选择算子：锦标赛
    toolbox.register("select", tools.selTournament, tournsize=3)
    # 交叉算子：适合排列的部分匹配交叉 (PMX)
    toolbox.register("mate", tools.cxPartialyMatched)
    # 变异算子：随机交换两个基因
    def mutate_shuffle(individual, indpb=0.05):
        tools.mutShuffleIndexes(individual, indpb=indpb)
        return (individual,)
    toolbox.register("mutate", mutate_shuffle, indpb=0.05)

    # --------------------------
    #  3. 运行遗传算法
    # --------------------------
    population_size = 50
    cxpb = 0.7         # 交叉概率
    mutpb = 0.2        # 变异概率
    n_generations = 200

    # 初始化种群
    population = toolbox.population(n=population_size)

    # 统计信息
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", max)
    stats.register("avg", lambda vs: sum(vs) / len(vs))

    # 最优解记录器
    hof = tools.HallOfFame(1)

    # 运行进化
    population, log = algorithms.eaSimple(population, toolbox,
                                          cxpb=cxpb, mutpb=mutpb,
                                          ngen=n_generations,
                                          stats=stats,
                                          halloffame=hof,
                                          verbose=True)

    # --------------------------
    #  4. 输出结果
    # --------------------------
    best_ind = hof[0]
    best_score = best_ind.fitness.values[0]
    max_possible = 2 * size * (size - 1)
    print(f"进化结束，最佳个体的匹配数为 {best_score} / {max_possible}")
    # 构造并打印网格
    solution_grid = arrangement_to_grid(best_ind, size)
    print("最终解的网格布局：")
    print_grid(solution_grid)


if __name__ == "__main__":
    main()
