# 介绍

- [四邻游戏](https://help.gnome.org/users/gnome-tetravex/unstable/index.html.zh_CN)是一个简单的解谜游戏，目标是把右侧的方块全部移动到左侧，每个方块的四面都邻接着相同的数字。
- 本项目通过自己实现回溯算法，以及使用python-constraint库来实现四邻游戏的自动解答
- utils.py提供了游戏输入创建和方块类；silin_solve.py提供了回溯算法实现；silin_solve_use_python_constraint.py提供了使用python-constraint库的自动解答实现