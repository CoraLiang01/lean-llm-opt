import gurobipy as gp
from gurobipy import GRB
clinics = ['K1', 'K2', 'K3', 'K4', 'K5', 'K6']
neighborhoods = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10']
demand = {'N1': 30, 'N2': 45, 'N3': 25, 'N4': 50, 'N5': 40, 'N6': 35, 'N7': 55, 'N8': 20, 'N9': 60, 'N10': 30}
distance = {'K1': {'N1': 2, 'N2': 3, 'N3': 9, 'N4': 10, 'N5': 11, 'N6': 12, 'N7': 13, 'N8': 14, 'N9': 15, 'N10': 16}, 'K2': {'N1': 3, 'N2': 2, 'N3': 8, 'N4': 9, 'N5': 10, 'N6': 11, 'N7': 12, 'N8': 13, 'N9': 14, 'N10': 15}, 'K3': {'N1': 10, 'N2': 9, 'N3': 2, 'N4': 3, 'N5': 4, 'N6': 9, 'N7': 10, 'N8': 11, 'N9': 12, 'N10': 13}, 'K4': {'N1': 11, 'N2': 10, 'N3': 3, 'N4': 2, 'N5': 5, 'N6': 8, 'N7': 9, 'N8': 10, 'N9': 11, 'N10': 12}, 'K5': {'N1': 13, 'N2': 12, 'N3': 10, 'N4': 9, 'N5': 8, 'N6': 2, 'N7': 3, 'N8': 4, 'N9': 8, 'N10': 9}, 'K6': {'N1': 14, 'N2': 13, 'N3': 11, 'N4': 10, 'N5': 9, 'N6': 3, 'N7': 2, 'N8': 5, 'N9': 3, 'N10': 2}}
p = 3
for i in clinics:
    if i not in distance or set(distance[i].keys()) != set(neighborhoods):
        raise ValueError(f'Distance data missing for clinic {i} or its neighborhoods.')
if set(demand.keys()) != set(neighborhoods):
    raise ValueError('Demand data missing for some neighborhoods.')
m = gp.Model('p_median_clinic_location')
x_vars = m.addVars(clinics, neighborhoods, vtype=GRB.BINARY, name='')
y_vars = m.addVars(clinics, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((demand[j] * distance[i][j] * x_vars[i, j] for i in clinics for j in neighborhoods)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in clinics)) == 1 for j in neighborhoods), name='')
m.addConstr(gp.quicksum((y_vars[i] for i in clinics)) == p, name='open_p')
m.addConstrs((x_vars[i, j] <= y_vars[i] for i in clinics for j in neighborhoods), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')