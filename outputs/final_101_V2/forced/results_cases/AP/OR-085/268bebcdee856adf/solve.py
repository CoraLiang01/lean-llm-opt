import gurobipy as gp
from gurobipy import GRB
locations = list(range(1, 16))
d = {1: {1: 0, 2: 67, 3: 55, 4: 80, 5: 21, 6: 77, 7: 78, 8: 74, 9: 85, 10: 28, 11: 55, 12: 53, 13: 66, 14: 89, 15: 78}, 2: {1: 67, 2: 0, 3: 38, 4: 29, 5: 68, 6: 36, 7: 62, 8: 54, 9: 49, 10: 92, 11: 37, 12: 51, 13: 38, 14: 82, 15: 31}, 3: {1: 55, 2: 38, 3: 0, 4: 28, 5: 44, 6: 27, 7: 56, 8: 34, 9: 33, 10: 68, 11: 70, 12: 55, 13: 46, 14: 32, 15: 40}, 4: {1: 80, 2: 29, 3: 28, 4: 0, 5: 21, 6: 51, 7: 46, 8: 48, 9: 31, 10: 55, 11: 68, 12: 85, 13: 58, 14: 56, 15: 22}, 5: {1: 21, 2: 68, 3: 44, 4: 21, 5: 0, 6: 42, 7: 57, 8: 31, 9: 55, 10: 79, 11: 49, 12: 70, 13: 43, 14: 55, 15: 78}, 6: {1: 77, 2: 36, 3: 27, 4: 51, 5: 42, 6: 0, 7: 63, 8: 41, 9: 39, 10: 52, 11: 76, 12: 54, 13: 59, 14: 44, 15: 76}, 7: {1: 78, 2: 62, 3: 56, 4: 46, 5: 57, 6: 63, 7: 0, 8: 38, 9: 35, 10: 37, 11: 55, 12: 54, 13: 51, 14: 14, 15: 64}, 8: {1: 74, 2: 54, 3: 34, 4: 48, 5: 31, 6: 41, 7: 38, 8: 0, 9: 53, 10: 24, 11: 60, 12: 42, 13: 31, 14: 42, 15: 27}, 9: {1: 85, 2: 49, 3: 33, 4: 31, 5: 55, 6: 39, 7: 35, 8: 53, 9: 0, 10: 88, 11: 28, 12: 65, 13: 12, 14: 63, 15: 45}, 10: {1: 28, 2: 92, 3: 68, 4: 55, 5: 79, 6: 52, 7: 37, 8: 24, 9: 88, 10: 0, 11: 84, 12: 40, 13: 43, 14: 81, 15: 37}, 11: {1: 55, 2: 37, 3: 70, 4: 68, 5: 49, 6: 76, 7: 55, 8: 60, 9: 28, 10: 84, 11: 0, 12: 41, 13: 38, 14: 56, 15: 35}, 12: {1: 53, 2: 51, 3: 55, 4: 85, 5: 70, 6: 54, 7: 54, 8: 42, 9: 65, 10: 40, 11: 41, 12: 0, 13: 65, 14: 47, 15: 38}, 13: {1: 66, 2: 38, 3: 46, 4: 58, 5: 43, 6: 59, 7: 51, 8: 31, 9: 12, 10: 43, 11: 38, 12: 65, 13: 0, 14: 35, 15: 77}, 14: {1: 89, 2: 82, 3: 32, 4: 56, 5: 55, 6: 44, 7: 14, 8: 42, 9: 63, 10: 81, 11: 56, 12: 47, 13: 35, 14: 0, 15: 54}, 15: {1: 78, 2: 31, 3: 40, 4: 22, 5: 78, 6: 76, 7: 64, 8: 27, 9: 45, 10: 37, 11: 35, 12: 38, 13: 77, 14: 54, 15: 0}}
for i in locations:
    for j in locations:
        if i == j:
            if d[i][j] != 0:
                raise ValueError(f'd[{i}][{j}] diagonal must be zero')
        else:
            if j not in d[i]:
                raise ValueError(f'd[{i}][{j}] missing')
            if i not in d[j]:
                raise ValueError(f'd[{j}][{i}] missing')
            if d[i][j] != d[j][i]:
                raise ValueError(f'd[{i}][{j}] != d[{j}][{i}]')
pairs = [(i, j) for i in locations for j in locations if i != j]
m = gp.Model('TSP_15')
x = m.addVars(pairs, vtype=GRB.BINARY, name='')
u_indices = list(range(2, 16))
u = m.addVars(u_indices, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((d[i][j] * x[i, j] for i, j in pairs)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in locations if j != i)) == 1 for i in locations), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in locations if i != j)) == 1 for j in locations), name='')
for i in u_indices:
    for j in u_indices:
        if i != j:
            m.addConstr(u[i] - u[j] + 15 * x[i, j] <= 14, name=f'mtz_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')