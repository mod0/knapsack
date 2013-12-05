#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import pdb

BB_solution = None
BB_stack = []
BB_tree = []

class BTreeNode():
    """Structure to represent  a binary tree node in branch and bound method""" 
    def __init__(self):
        self.ParentNode = None
        #self.LeftNode = None
        #self.RightNode = None
        self.Relaxation = 0
        self.Objective = 0
        self.ObjectID = -1
        self.Taken = None
        self.Room = None

    def __str__(self):
        return str(self.ObjectID) + "[" + str(self.Taken) + "]:" + str(self.Objective) +  ", " + str(self.Room) + ", " + str(self.Relaxation)

def DP_solver(capacity, weights, values):
    taken = [] 
    dp_table = [];

    items = len(values)

    # Allocate memory for dp_table
    for j in xrange(0, capacity + 1):
        dp_table.append([])
        for i in xrange(0, items + 1):
            dp_table[j].append(0)

    # Allocate memory for taken
    for i in xrange(0, items):
        taken.append(0)

    # Solve DP
    for i in xrange(0, items + 1):
        for j in xrange(0, capacity + 1):
            if i == 0 or j == 0:
                dp_table[j][i] = 0
            elif weights[i - 1] <= j:
                dp_table[j][i] = max(dp_table[j][i - 1], values[i - 1] + dp_table[j - weights[i - 1]][i - 1])
            else:
                dp_table[j][i] = dp_table[j][i - 1]

    # Backtrace step
    j = capacity + 1
    for i in xrange(items, 0, -1):
        if dp_table[j - 1][i] > dp_table[j - 1][i - 1]:
            taken[i - 1] = 1
            j = j - weights[i - 1]

    value = dp_table[capacity][items]

    return (value, taken)


def BB_solver(capacity, weights, values):
    # array indicating whether an 
    # element has been taken or not.
    global BB_tree, BB_stack

    # Get number of items
    items = len(values)

    # Allocate memory for taken
    taken = [0] * items

    # Create a list containing (index, value/weight)
    value_per_weight = [(elem[0][0], elem[0][1]/elem[1]) for elem in zip(enumerate(values), weights)]

    # Sort the list in descending order
    value_per_weight.sort(key=lambda pair:pair[1], reverse=True)

    # Reorder values and weights
    weights = [weights[element[0]] for element in value_per_weight]
    values = [values[element[0]] for element in value_per_weight]

    # Create root node
    Root = BTreeNode()
    #Root.Relaxation = sum(values)
    Root.Room = capacity
    Root.Objective = 0
    Root.ObjectID = -1
    Root.Relaxation = getBound(items - 1, Root.ObjectID, Root.Room, Root.Objective, weights, values, value_per_weight)

    # Add root node to the tree
    BB_tree.append(Root)
    BB_stack.append(Root)

    # Branch while stack is not empty
    while BB_stack:
        Branch(items - 1, values, weights, value_per_weight)

    # Retrace which items were taken and which ignored
    Node = BB_solution

    while Node.ParentNode:
        taken[value_per_weight[Node.ObjectID][0]] = Node.Taken
        Node = Node.ParentNode

    return (BB_solution.Objective, taken)


def Branch(items, values, weights, value_per_weight):
    global BB_solution, BB_tree, BB_stack

    if not BB_stack:
        return
    else:
        Root = BB_stack.pop()

    if BB_solution and Root.Relaxation < BB_solution.Objective:
        return
    elif Root.ObjectID == items:
        return

    Node = BTreeNode()
    Node.ObjectID = Root.ObjectID + 1
    Node.Room = Root.Room
    if Node.Room >= 0:
        Node.Objective = Root.Objective 
        Node.Taken = 0
        #Node.Relaxation = Root.Relaxation - values[Node.ObjectID]
        Node.Relaxation = getBound(items, Node.ObjectID, Node.Room, Node.Objective, weights, values, value_per_weight)
        Node.ParentNode = Root
        #Root.RightNode = Node
        BB_stack.append(Node)
        if Node.Objective == Node.Relaxation:
            if BB_solution and Node.Objective > BB_solution.Objective:
                BB_solution = Node
            elif BB_solution is None:
                BB_solution = Node

    BB_tree.append(Node)

    Node = BTreeNode()
    Node.ObjectID = Root.ObjectID + 1
    Node.Room = Root.Room - weights[Node.ObjectID]
    if Node.Room >= 0:
        Node.Objective = Root.Objective + values[Node.ObjectID]
        Node.Taken = 1
        #Node.Relaxation = Root.Relaxation
        Node.Relaxation = getBound(items, Node.ObjectID, Node.Room, Node.Objective, weights, values, value_per_weight)
        Node.ParentNode = Root
        #Root.LeftNode = Node
        BB_stack.append(Node)
        if Node.Objective == Node.Relaxation:
            if BB_solution and Node.Objective > BB_solution.Objective:
                BB_solution = Node
            elif BB_solution is None:
                BB_solution = Node
    
    BB_tree.append(Node)


def BB_solver_lm(capacity, weights, values):
    # array indicating whether an 
    # element has been taken or not.
    global BB_stack

    # Get number of items
    items = len(values)

    # Create a list containing (index, value/weight)
    value_per_weight = [(elem[0][0], elem[0][1]/elem[1]) for elem in zip(enumerate(values), weights)]

    # Sort the list in descending order
    value_per_weight.sort(key=lambda pair:pair[1], reverse=True)

    # Reorder values and weights
    weights = [weights[element[0]] for element in value_per_weight]
    values = [values[element[0]] for element in value_per_weight]

    # Create root node
    Root = BTreeNode()
    # Root.Relaxation = sum(values)
    Root.Room = capacity
    Root.Objective = 0
    Root.ObjectID = -1
    Root.Relaxation = getBound(items - 1, Root.ObjectID, Root.Room, Root.Objective, weights, values, value_per_weight)
    Root.Taken = [0] * items

    # Add root node to the stack
    BB_stack.append(Root)

    # Branch on the root node by taking the next node
    while BB_stack:
        Branch_lm(items - 1, values, weights, value_per_weight)

    # Remap taken items
    taken = [0] * items
    for index, element in enumerate(value_per_weight):
        taken[element[0]] = BB_solution.Taken[index]

    return (BB_solution.Objective, taken)


def Branch_lm(items, values, weights, value_per_weight):
    global BB_solution, BB_stack

    if not BB_stack:
        return
    else:
        Root = BB_stack.pop()

    if BB_solution and Root.Relaxation < BB_solution.Objective:
        return
    elif Root.ObjectID == items:
        return

    Node = BTreeNode()
    Node.ObjectID = Root.ObjectID + 1
    Node.Room = Root.Room
    if Node.Room >= 0:
        Node.Objective = Root.Objective 
        Node.Taken = Root.Taken
        # Node.Relaxation = Root.Relaxation - values[Node.ObjectID]
        Node.Relaxation = getBound(items, Node.ObjectID, Node.Room, Node.Objective, weights, values, value_per_weight)
        BB_stack.append(Node)
        if Node.Objective == Node.Relaxation:
            if BB_solution and Node.Objective > BB_solution.Objective:
                BB_solution = Node
            elif BB_solution is None:
                BB_solution = Node

    Node = BTreeNode()
    Node.ObjectID = Root.ObjectID + 1
    Node.Room = Root.Room - weights[Node.ObjectID]
    if Node.Room >= 0:
        Node.Objective = Root.Objective + values[Node.ObjectID]    
        Node.Taken = list(Root.Taken)
        Node.Taken[Node.ObjectID] = 1
        # Node.Relaxation = Root.Relaxation
        Node.Relaxation = getBound(items, Node.ObjectID, Node.Room, Node.Objective, weights, values, value_per_weight)
        BB_stack.append(Node)
        if Node.Objective == Node.Relaxation:
            if BB_solution and Node.Objective > BB_solution.Objective:
                BB_solution = Node
            elif BB_solution is None:
                BB_solution = Node


def getBound(items, rootid, root_room, root_objective, weights, values, value_per_weight):
    while rootid < items and root_room - weights[rootid + 1] >= 0:
        root_objective = root_objective + values[rootid + 1]
        root_room = root_room - weights[rootid + 1]
        rootid = rootid + 1

    if rootid < items and root_room > 0:
        root_objective = root_objective + min(root_room, weights[rootid + 1]) * value_per_weight[rootid + 1][1]

    return root_objective


def solveIt(inputData):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = inputData.split('\n')

    firstLine = lines[0].split()
    items = int(firstLine[0])
    capacity = int(firstLine[1])

    values = []
    weights = []

    for i in range(1, items+1):
        line = lines[i]
        parts = line.split()

        values.append(int(parts[0]))
        weights.append(int(parts[1]))

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    # value, taken = DP_solver(capacity, weights, values)
    value , taken = BB_solver(capacity, weights, values)
    # value, taken = BB_solver_lm(capacity, weights, values)

    # prepare the solution in the specified output format
    outputData = str(value) + ' ' + str(0) + '\n'
    outputData += ' '.join(map(str, taken))
    return outputData


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, 'r')
        inputData = ''.join(inputDataFile.readlines())
        inputDataFile.close()
        print solveIt(inputData)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'