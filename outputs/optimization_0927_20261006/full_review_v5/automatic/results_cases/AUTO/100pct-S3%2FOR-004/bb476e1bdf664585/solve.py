import gurobipy as gp
from gurobipy import GRB
machines = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']
tasks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
cost = {'M1': {'A': 167.4, 'B': 98.6, 'C': 189.4, 'D': 119.6, 'E': 182.0, 'F': 145.1, 'G': 185.4, 'H': 94.8, 'I': 122.3, 'J': 123.3, 'K': 96.1, 'L': 90.3}, 'M2': {'A': 156.2, 'B': 88.7, 'C': 187.3, 'D': 124.7, 'E': 173.2, 'F': 144.3, 'G': 179.0, 'H': 91.5, 'I': 115.1, 'J': 119.5, 'K': 100.1, 'L': 88.6}, 'M3': {'A': 184.3, 'B': 121.0, 'C': 216.6, 'D': 140.0, 'E': 196.2, 'F': 168.8, 'G': 205.6, 'H': 114.2, 'I': 133.3, 'J': 144.5, 'K': 116.0, 'L': 107.7}, 'M4': {'A': 157.9, 'B': 92.9, 'C': 185.1, 'D': 120.3, 'E': 175.1, 'F': 146.2, 'G': 180.8, 'H': 86.3, 'I': 111.6, 'J': 115.9, 'K': 98.1, 'L': 91.1}, 'M5': {'A': 175.6, 'B': 103.6, 'C': 204.5, 'D': 130.0, 'E': 192.8, 'F': 157.5, 'G': 194.2, 'H': 106.9, 'I': 129.9, 'J': 134.9, 'K': 105.8, 'L': 98.6}, 'M6': {'A': 166.8, 'B': 107.0, 'C': 199.2, 'D': 130.4, 'E': 183.6, 'F': 159.5, 'G': 187.0, 'H': 98.2, 'I': 121.3, 'J': 126.2, 'K': 105.9, 'L': 101.8}, 'M7': {'A': 159.7, 'B': 93.2, 'C': 183.8, 'D': 113.0, 'E': 171.9, 'F': 139.1, 'G': 169.6, 'H': 85.1, 'I': 110.0, 'J': 116.7, 'K': 90.6, 'L': 85.2}, 'M8': {'A': 184.8, 'B': 115.9, 'C': 205.1, 'D': 138.6, 'E': 195.4, 'F': 160.1, 'G': 200.2, 'H': 108.5, 'I': 136.9, 'J': 140.0, 'K': 114.6, 'L': 103.9}, 'M9': {'A': 157.3, 'B': 86.2, 'C': 186.0, 'D': 113.9, 'E': 166.2, 'F': 136.8, 'G': 167.5, 'H': 78.8, 'I': 107.4, 'J': 114.5, 'K': 87.2, 'L': 78.6}, 'M10': {'A': 164.8, 'B': 97.8, 'C': 200.9, 'D': 125.8, 'E': 188.9, 'F': 151.2, 'G': 187.7, 'H': 99.5, 'I': 119.5, 'J': 132.1, 'K': 101.1, 'L': 98.4}, 'M11': {'A': 164.0, 'B': 92.2, 'C': 186.2, 'D': 115.7, 'E': 174.5, 'F': 143.0, 'G': 175.9, 'H': 92.3, 'I': 114.0, 'J': 121.2, 'K': 93.7, 'L': 91.2}, 'M12': {'A': 151.7, 'B': 76.7, 'C': 179.5, 'D': 109.5, 'E': 160.6, 'F': 128.4, 'G': 170.2, 'H': 74.4, 'I': 103.7, 'J': 110.4, 'K': 83.7, 'L': 75.2}}
for i in machines:
    if i not in cost or not isinstance(cost[i], dict):
        raise ValueError(f'Missing cost row for machine {i}')
    for j in tasks:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for machine {i}, task {j}')
m = gp.Model('factory_assignment')
x_vars = m.addVars(machines, tasks, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in machines for j in tasks)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for j in tasks)) == 1 for i in machines), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for i in machines)) == 1 for j in tasks), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')