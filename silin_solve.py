from typing import List, Optional
from utils import Cube, PuzzleGenerator
import copy


class PuzzleSolver:
    def __init__(self, cubes: List[Cube], size: int = 3):
        self.size = size
        self.cubes = cubes.copy()  # 剩余未放置的方块
        self.grid: List[List[Optional[Cube]]] = [
            [None for _ in range(size)] for _ in range(size)
        ]
        self.solutions: List[List[List[Cube]]] = []

    def is_solution(self, state: List[List[Optional[Cube]]]) -> bool:
        """
        判断当前状态是否为解，即所有位置都已被放置方块。
        """
        for row in state:
            if any(cube is None for cube in row):
                return False
        return True

    def record_solution(self, state: List[List[Optional[Cube]]]):
        """
        记录当前解的拷贝。
        """
        solution_copy = copy.deepcopy(state)
        self.solutions.append(solution_copy)

    def is_valid(self, row: int, col: int, cube: Cube) -> bool:
        """
        判断在 (row, col) 放置 cube 是否合法。
        """
        # 检查上方方块的 'down' 是否匹配
        if row > 0 and self.grid[row - 1][col] is not None:
            if self.grid[row - 1][col].down != cube.up:
                return False

        # 检查左侧方块的 'right' 是否匹配
        if col > 0 and self.grid[row][col - 1] is not None:
            if self.grid[row][col - 1].right != cube.left:
                return False

        # 检查下方方块的 'up' 是否匹配（如果下方已经放置）
        if row < self.size - 1 and self.grid[row + 1][col] is not None:
            if self.grid[row + 1][col].up != cube.down:
                return False

        # 检查右侧方块的 'left' 是否匹配（如果右侧已经放置）
        if col < self.size - 1 and self.grid[row][col + 1] is not None:
            if self.grid[row][col + 1].left != cube.right:
                return False

        return True

    def make_choice(self, row: int, col: int, cube: Cube):
        """
        在 (row, col) 放置 cube，并从剩余方块中移除它。
        """
        self.grid[row][col] = cube
        self.cubes.remove(cube)

    def undo_choice(self, row: int, col: int, cube: Cube):
        """
        从 (row, col) 移除 cube，并将其添加回剩余方块中。
        """
        self.grid[row][col] = None
        self.cubes.append(cube)

    def backtrack(self, row: int, col: int):
        """
        回溯搜索函数，尝试在 (row, col) 放置方块。
        """
        # 如果已经处理完所有行，记录解
        if row == self.size:
            self.record_solution(self.grid)
            return

        # 计算下一个位置
        next_row, next_col = (row, col + 1) if (col + 1) < self.size else (row + 1, 0)

        for cube in self.cubes.copy():
            if self.is_valid(row, col, cube):
                self.make_choice(row, col, cube)
                self.backtrack(next_row, next_col)
                self.undo_choice(row, col, cube)

    def solve(self):
        """
        启动回溯搜索，寻找所有可能的解。
        """
        self.backtrack(0, 0)
        if self.solutions:
            print(f"找到 {len(self.solutions)} 个解。展示第一个解：\n")
            self.print_solution(self.solutions[0])
        else:
            print("没有找到解。")

    def print_solution(self, solution: List[List[Optional[Cube]]]):
        """
        打印解的网格布局。
        """
        for i in range(self.size):
            # 打印 'up' 数字
            up_line = ""
            for j in range(self.size):
                up_line += f"  {solution[i][j].up}   "
            print(up_line)

            # 打印 'left' 和 'right' 数字
            middle_line = ""
            for j in range(self.size):
                middle_line += f"{solution[i][j].left} * {solution[i][j].right} "
            print(middle_line)

            # 打印 'down' 数字
            down_line = ""
            for j in range(self.size):
                down_line += f"  {solution[i][j].down}   "
            print(down_line)
            print("\n")


# 示例使用
if __name__ == "__main__":
    # 生成初始合法拼图
    generator = PuzzleGenerator(size=5, max_num=5)
    print("初始解的网格布局：\n")
    generator.print_initial_solution()
    all_cubes = generator.get_random_cubes()

    # 创建求解器并尝试求解
    solver = PuzzleSolver(cubes=all_cubes, size=5)
    solver.solve()
