# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)
import queue

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    start = maze.start
    endgoal = maze.waypoints[0]
    
    return_path = []
    explored_set = set()
    
    queue_maze = queue.Queue()
    
    cur_pos = start
    
    
    queue_maze.put(start)
    explored_set.add(start)
    
    prev_pos = {}
    
    while (queue_maze):
        cur_pos = queue_maze.get()
        if(cur_pos == endgoal):
            while(cur_pos != start):
                return_path.append(cur_pos)
                cur_pos = prev_pos[cur_pos]
            return_path.append(start)
            return_path.reverse()
            return return_path
        explored_set.add(cur_pos)
        for i in maze.neighbors_all(cur_pos[0], cur_pos[1]):
            if i not in explored_set:
                prev_pos[i] = cur_pos
                explored_set.add(i)
                queue_maze.put(i)
    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    start = maze.start
    endgoal = maze.waypoints[0]
    
    return_path = []
    explored_set = set()
    
    queue_maze = queue.PriorityQueue()
    queue_maze.put((0, start))
    prev_pos = {}
    prev_pos[start] = None

    g = {}
    g[start] = 0
    def h(a):
        return abs(a[0] - endgoal[0]) + abs(a[1] - endgoal[1])
    
    while (queue_maze):
        cur_pos = queue_maze.get()[1]
        if (cur_pos == endgoal):
            while endgoal != start:
                return_path.append(endgoal)
                endgoal = prev_pos[endgoal]
            return_path.append(start)
            return_path.reverse()
            return return_path
        explored_set.add(cur_pos)
        for i in maze.neighbors_all(cur_pos[0], cur_pos[1]):
            g_temp = g[cur_pos] + 1
            if i not in g or g_temp < g[i] or i not in explored_set:
                g[i] = g_temp
                explored_set.add(i)
                prev_pos[i] = cur_pos
                queue_maze.put((h(i) + g[i] - cur_pos[1] - cur_pos[0], i))
    
    return return_path

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
