from constraint import (
    Problem,
    AllDifferentConstraint,
    BacktrackingSolver,
    RecursiveBacktrackingSolver,
    MinConflictsSolver,
)
from typing import List
from utils import Cube, PuzzleGenerator


class PuzzleConstraintSolver:
    def __init__(
        self,
        cubes: List[Cube],
        size: int = 3,
        all_solutions=False,
        solver_class=BacktrackingSolver,
    ):
        self.size = size
        self.cubes = cubes
        self.all_solutions = all_solutions
        self.problem = Problem(solver=solver_class())

    def solve_puzzle(self):
        """
        Solves the puzzle using the constraint programming approach.
        """
        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        # Each position can take any cube, but all positions must have different cubes
        self.problem.addVariables(positions, self.cubes)
        self.problem.addConstraint(AllDifferentConstraint())

        # Define constraints for matching edges
        for i in range(self.size):
            for j in range(self.size):
                if i > 0:
                    # Current cube's up == above cube's down
                    self.problem.addConstraint(
                        lambda current, above: current.up == above.down,
                        ((i, j), (i - 1, j)),
                    )
                if j > 0:
                    # Current cube's left == left cube's right
                    self.problem.addConstraint(
                        lambda current, left_cube: current.left == left_cube.right,
                        ((i, j), (i, j - 1)),
                    )
        if self.all_solutions:
            solutions = self.problem.getSolutions()
            if solutions:
                print(f"找到 {len(solutions)} 个解。展示第一个解：\n")
                solution = solutions[0]
                self.print_solution(solution)
            else:
                print("没有找到解。")
        else:
            solution = self.problem.getSolution()
            if solution:
                self.print_solution(solution)
            else:
                print("没有找到解。")

    def print_solution(self, solution: dict):
        """
        Prints the solved grid.
        """
        grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        for position, cube in solution.items():
            grid[position[0]][position[1]] = cube

        for i in range(self.size):
            # Print the 'up' numbers
            up_line = ""
            for j in range(self.size):
                up_line += f"  {grid[i][j].up}   "
            print(up_line)

            # Print the 'left' and 'right' numbers
            middle_line = ""
            for j in range(self.size):
                middle_line += f"{grid[i][j].left} * {grid[i][j].right} "
            print(middle_line)

            # Print the 'down' numbers
            down_line = ""
            for j in range(self.size):
                down_line += f"  {grid[i][j].down}   "
            print(down_line)
            print("\n")


# 示例使用
if __name__ == "__main__":
    # 生成初始合法拼图
    board_size = 5
    generator = PuzzleGenerator(size=board_size, max_num=5)
    generator.print_initial_solution()
    all_cubes = generator.get_random_cubes()
    import time

    # 使用3种求解器来求解，并比较时间
    for solver_class in [
        BacktrackingSolver,
        RecursiveBacktrackingSolver,
        MinConflictsSolver,
    ]:
        # 创建 CSP 求解器并尝试求解
        print(f"使用 {solver_class.__name__} 求解器：")
        start_time = time.time()
        solver = PuzzleConstraintSolver(
            cubes=all_cubes,
            size=board_size,
            all_solutions=False,
            solver_class=solver_class,
        )
        solver.solve_puzzle()
        end_time = time.time()
        print(f"求解时间：{end_time - start_time} 秒")
