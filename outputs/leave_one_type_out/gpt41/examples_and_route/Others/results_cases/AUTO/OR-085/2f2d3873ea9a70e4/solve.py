import gurobipy as gp
from gurobipy import GRB
locations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
distance_matrix = {1: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}, 2: {1: 0, 3: 38, 4: 29, 5: 68, 6: 36, 7: 62, 8: 54, 9: 49, 10: 92, 11: 37, 12: 51, 13: 38, 14: 82, 15: 31}, 3: {1: 0, 2: 67, 4: 28, 5: 44, 6: 27, 7: 56, 8: 34, 9: 33, 10: 68, 11: 70, 12: 55, 13: 46, 14: 32, 15: 40}, 4: {1: 0, 2: 55, 3: 28, 5: 21, 6: 51, 7: 46, 8: 48, 9: 31, 10: 55, 11: 68, 12: 85, 13: 58, 14: 56, 15: 22}, 5: {1: 0, 2: 80, 3: 29, 4: 21, 6: 42, 7: 57, 8: 31, 9: 55, 10: 79, 11: 49, 12: 70, 13: 43, 14: 55, 15: 78}, 6: {1: 0, 2: 21, 3: 68, 4: 44, 5: 42, 7: 63, 8: 41, 9: 39, 10: 52, 11: 76, 12: 54, 13: 59, 14: 44, 15: 76}, 7: {1: 0, 2: 77, 3: 36, 4: 27, 5: 51, 6: 63, 8: 38, 9: 35, 10: 37, 11: 55, 12: 54, 13: 51, 14: 14, 15: 64}, 8: {1: 0, 2: 78, 3: 62, 4: 56, 5: 46, 6: 63, 7: 38, 9: 53, 10: 24, 11: 60, 12: 42, 13: 31, 14: 42, 15: 27}, 9: {1: 0, 2: 74, 3: 54, 4: 34, 5: 48, 6: 41, 7: 38, 8: 53, 10: 88, 11: 28, 12: 65, 13: 12, 14: 63, 15: 45}, 10: {1: 0, 2: 85, 3: 49, 4: 33, 5: 31, 6: 39, 7: 35, 8: 53, 9: 88, 11: 84, 12: 40, 13: 43, 14: 81, 15: 37}, 11: {1: 0, 2: 28, 3: 92, 4: 68, 5: 55, 6: 52, 7: 37, 8: 24, 9: 88, 10: 84, 12: 41, 13: 38, 14: 56, 15: 35}, 12: {1: 0, 2: 55, 3: 37, 4: 70, 5: 68, 6: 76, 7: 55, 8: 60, 9: 28, 10: 84, 11: 41, 13: 65, 14: 47, 15: 38}, 13: {1: 0, 2: 53, 3: 51, 4: 55, 5: 85, 6: 54, 7: 54, 8: 42, 9: 65, 10: 40, 11: 41, 12: 65, 14: 35, 15: 77}, 14: {1: 0, 2: 66, 3: 38, 4: 46, 5: 58, 6: 59, 7: 51, 8: 31, 9: 12, 10: 43, 11: 38, 12: 65, 13: 35, 15: 54}, 15: {1: 0, 2: 89, 3: 82, 4: 32, 5: 56, 6: 44, 7: 14, 8: 42, 9: 63, 10: 56, 11: 47, 12: 35, 13: 54, 14: 54}}
for i in locations:
    if i not in distance_matrix:
        distance_matrix[i] = {}
    for j in locations:
        if i == j:
            distance_matrix[i][j] = 0
        elif j not in distance_matrix[i]:
            if i in distance_matrix.get(j, {}):
                distance_matrix[i][j] = distance_matrix[j][i]
            else:
                raise ValueError(f'Missing distance between {i} and {j}')
for i in locations:
    for j in locations:
        if i != j and (j not in distance_matrix[i] or distance_matrix[i][j] == '' or distance_matrix[i][j] is None):
            raise ValueError(f'Missing distance between {i} and {j}')
m = gp.Model('TSP_15')
m.Params.MIPGap = 0.0001
x = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=GRB.BINARY, name='')
u = m.addVars(locations, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((distance_matrix[i][j] * x[i, j] for i in locations for j in locations if i != j)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in locations if j != i)) == 1 for i in locations), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in locations if i != j)) == 1 for j in locations), name='')
m.addConstr(u[1] == 1, name='u1fix')
for i in locations:
    if i == 1:
        continue
    m.addConstr(u[i] >= 2, name=f'u_lb_{i}')
    m.addConstr(u[i] <= 15, name=f'u_ub_{i}')
for i in locations:
    if i == 1:
        continue
    for j in locations:
        if j == 1 or i == j:
            continue
        m.addConstr(u[i] - u[j] + 15 * x[i, j] <= 14, name=f'mtz_{i}_{j}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')