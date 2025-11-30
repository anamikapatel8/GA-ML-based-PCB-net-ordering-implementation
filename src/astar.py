import heapq
import math
import numpy as np

def astar(start, goal, grid):
    """A* on 2D numpy grid. 0=free, 1=blocked"""
    rows, cols = grid.shape
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get((nx,ny), math.inf):
                    g_score[(nx,ny)] = tentative_g
                    f = tentative_g + heuristic((nx,ny), goal)
                    heapq.heappush(open_heap, (f, (nx,ny)))
                    came_from[(nx,ny)] = current
    return None
