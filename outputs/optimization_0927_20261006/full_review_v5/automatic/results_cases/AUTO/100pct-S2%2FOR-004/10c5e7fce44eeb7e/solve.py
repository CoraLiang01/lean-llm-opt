import gurobipy as gp
from gurobipy import GRB
machines = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']
tasks = ['Project_A', 'Project_B', 'Project_C', 'Project_D', 'Project_E', 'Project_F', 'Project_G', 'Project_H', 'Project_I', 'Project_J', 'Project_K', 'Project_L']
cost = {'M1': {'Project_A': 167.4, 'Project_B': 98.6, 'Project_C': 189.4, 'Project_D': 119.6, 'Project_E': 182.0, 'Project_F': 145.1, 'Project_G': 185.4, 'Project_H': 94.8, 'Project_I': 122.3, 'Project_J': 123.3, 'Project_K': 96.1, 'Project_L': 90.3}, 'M2': {'Project_A': 156.2, 'Project_B': 88.7, 'Project_C': 187.3, 'Project_D': 124.7, 'Project_E': 173.2, 'Project_F': 144.3, 'Project_G': 179.0, 'Project_H': 91.5, 'Project_I': 115.1, 'Project_J': 119.5, 'Project_K': 100.1, 'Project_L': 88.6}, 'M3': {'Project_A': 184.3, 'Project_B': 121.0, 'Project_C': 216.6, 'Project_D': 140.0, 'Project_E': 196.2, 'Project_F': 168.8, 'Project_G': 205.6, 'Project_H': 114.2, 'Project_I': 133.3, 'Project_J': 144.5, 'Project_K': 116.0, 'Project_L': 107.7}, 'M4': {'Project_A': 157.9, 'Project_B': 92.9, 'Project_C': 185.1, 'Project_D': 120.3, 'Project_E': 175.1, 'Project_F': 146.2, 'Project_G': 180.8, 'Project_H': 86.3, 'Project_I': 111.6, 'Project_J': 115.9, 'Project_K': 98.1, 'Project_L': 91.1}, 'M5': {'Project_A': 175.6, 'Project_B': 103.6, 'Project_C': 204.5, 'Project_D': 130.0, 'Project_E': 192.8, 'Project_F': 157.5, 'Project_G': 194.2, 'Project_H': 106.9, 'Project_I': 129.9, 'Project_J': 134.9, 'Project_K': 105.8, 'Project_L': 98.6}, 'M6': {'Project_A': 166.8, 'Project_B': 107.0, 'Project_C': 199.2, 'Project_D': 130.4, 'Project_E': 183.6, 'Project_F': 159.5, 'Project_G': 187.0, 'Project_H': 98.2, 'Project_I': 121.3, 'Project_J': 126.2, 'Project_K': 105.9, 'Project_L': 101.8}, 'M7': {'Project_A': 159.7, 'Project_B': 93.2, 'Project_C': 183.8, 'Project_D': 113.0, 'Project_E': 171.9, 'Project_F': 139.1, 'Project_G': 169.6, 'Project_H': 85.1, 'Project_I': 110.0, 'Project_J': 116.7, 'Project_K': 90.6, 'Project_L': 85.2}, 'M8': {'Project_A': 184.8, 'Project_B': 115.9, 'Project_C': 205.1, 'Project_D': 138.6, 'Project_E': 195.4, 'Project_F': 160.1, 'Project_G': 200.2, 'Project_H': 108.5, 'Project_I': 136.9, 'Project_J': 140.0, 'Project_K': 114.6, 'Project_L': 103.9}, 'M9': {'Project_A': 157.3, 'Project_B': 86.2, 'Project_C': 186.0, 'Project_D': 113.9, 'Project_E': 166.2, 'Project_F': 136.8, 'Project_G': 167.5, 'Project_H': 78.8, 'Project_I': 107.4, 'Project_J': 114.5, 'Project_K': 87.2, 'Project_L': 78.6}, 'M10': {'Project_A': 164.8, 'Project_B': 97.8, 'Project_C': 200.9, 'Project_D': 125.8, 'Project_E': 188.9, 'Project_F': 151.2, 'Project_G': 187.7, 'Project_H': 99.5, 'Project_I': 119.5, 'Project_J': 132.1, 'Project_K': 101.1, 'Project_L': 98.4}, 'M11': {'Project_A': 164.0, 'Project_B': 92.2, 'Project_C': 186.2, 'Project_D': 115.7, 'Project_E': 174.5, 'Project_F': 143.0, 'Project_G': 175.9, 'Project_H': 92.3, 'Project_I': 114.0, 'Project_J': 121.2, 'Project_K': 93.7, 'Project_L': 91.2}, 'M12': {'Project_A': 151.7, 'Project_B': 76.7, 'Project_C': 179.5, 'Project_D': 109.5, 'Project_E': 160.6, 'Project_F': 128.4, 'Project_G': 170.2, 'Project_H': 74.4, 'Project_I': 103.7, 'Project_J': 110.4, 'Project_K': 83.7, 'Project_L': 75.2}}
for i in machines:
    if i not in cost or not isinstance(cost[i], dict):
        raise ValueError(f'Missing cost row for machine {i}')
    for j in tasks:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for machine {i}, task {j}')
m = gp.Model('Factory_Machine_Task_Assignment')
x_vars = m.addVars(machines, tasks, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in machines for j in tasks)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for j in tasks)) == 1 for i in machines), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for i in machines)) == 1 for j in tasks), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')