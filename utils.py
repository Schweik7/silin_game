from dataclasses import dataclass
import random


@dataclass
class Cube:
    up: int
    down: int
    left: int
    right: int

    def __hash__(self):
        return 1000 * self.up + 100 * self.down + 10 * self.left + self.right

    def __repr__(self):
        return f"""*{self.up}* 
{self.left}*{self.right}
*{self.down}*"""


class PuzzleGenerator:
    def __init__(self, size=3, max_num=9):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.cubes = []
        self.max_num = max_num
        self.generate_puzzle()

    def generate_puzzle(self):
        """
        Generates a grid with matching numbers and populates the cubes list.
        """
        for i in range(self.size):
            for j in range(self.size):
                # Determine the 'up' value
                if i == 0:
                    up = random.randint(0, self.max_num)
                else:
                    up = self.grid[i - 1][j].down

                # Determine the 'left' value
                if j == 0:
                    left = random.randint(0, self.max_num)
                else:
                    left = self.grid[i][j - 1].right

                # Randomly generate 'down' and 'right' values
                down = random.randint(0, self.max_num)
                right = random.randint(0, self.max_num)

                cube = Cube(up, down, left, right)
                self.grid[i][j] = cube
                self.cubes.append(cube)

        # After initial assignment, ensure consistency for the bottom and right edges
        # (This is already handled by setting 'up' and 'left' based on neighbors)

    def print_initial_solution(self):
        """
        Prints the grid of cubes in a readable rectangular format.
        """
        for i in range(self.size):
            # Print the 'up' numbers
            up_line = ""
            for j in range(self.size):
                up_line += f"  {self.grid[i][j].up}   "
            print(up_line)

            # Print the 'left' and 'right' numbers
            middle_line = ""
            for j in range(self.size):
                middle_line += f"{self.grid[i][j].left} * {self.grid[i][j].right} "
            print(middle_line)

            # Print the 'down' numbers
            down_line = ""
            for j in range(self.size):
                down_line += f"  {self.grid[i][j].down}   "
            print(down_line)
            print("\n")

    def get_cubes(self):
        """
        Returns the list of cubes.
        """
        return self.cubes

    def get_random_cubes(self):
        """
        Returns cubes with randomized order.
        """
        random.shuffle(self.cubes)
        return self.cubes


# 示例使用
if __name__ == "__main__":
    generator = PuzzleGenerator(size=6)
    print("初始解的网格布局：\n")
    generator.print_initial_solution()
