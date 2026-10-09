import gurobipy as gp
from gurobipy import GRB
centres = ['SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8', 'SC9', 'SC10']
customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
fixed_cost = {'SC1': 385.1, 'SC2': 546.3, 'SC3': 485.2, 'SC4': 448.1, 'SC5': 324.1, 'SC6': 323.9, 'SC7': 296.5, 'SC8': 522.7, 'SC9': 448.7, 'SC10': 478.7}
service_cost = {'SC1': [15.1, 13.4, 15.2, 16.8, 13.4, 12.5, 12.1, 12.3, 16.3, 12.1, 16.7, 11.3, 15.1, 8.3, 12.1], 'SC2': [21.2, 16.3, 18.8, 19.1, 18.6, 22.5, 17.1, 15.7, 21.3, 18.7, 18.7, 23.8, 20.5, 20.7, 16.3], 'SC3': [14.9, 20.2, 14.7, 18.3, 20.8, 15.5, 19.8, 17.9, 17.6, 14.4, 15.7, 15.5, 15.1, 14.7, 16.4], 'SC4': [18.8, 19.6, 21.7, 18.8, 19.8, 14.9, 18.6, 21.3, 20.8, 20.1, 19.9, 17.3, 18.4, 20.4, 15.1], 'SC5': [22.9, 20.9, 18.1, 23.1, 22.1, 21.6, 22.1, 22.7, 21.8, 22.7, 24.2, 23.2, 20.6, 20.6, 21.3], 'SC6': [16.8, 22.1, 18.6, 15.7, 18.1, 21.3, 20.7, 15.3, 17.2, 14.1, 18.7, 17.7, 17.9, 14.8, 19.1], 'SC7': [16.5, 16.9, 12.3, 13.1, 16.7, 16.1, 20.5, 16.6, 15.5, 18.1, 14.2, 16.8, 14.5, 14.2, 19.5], 'SC8': [9.4, 9.4, 11.2, 8.6, 12.1, 10.7, 12.2, 11.4, 12.6, 11.4, 13.1, 14.5, 8.5, 11.5, 16.7], 'SC9': [16.1, 13.8, 11.9, 15.6, 11.4, 11.9, 15.4, 14.1, 19.9, 18.1, 14.7, 15.8, 14.9, 14.1, 11.1], 'SC10': [17.3, 11.7, 20.4, 22.2, 18.2, 14.6, 18.7, 20.1, 19.1, 17.4, 16.1, 17.8, 13.9, 15.1, 18.7]}
if set(fixed_cost.keys()) != set(centres):
    raise ValueError('Fixed cost data missing for some centres.')
for i in centres:
    if i not in service_cost or len(service_cost[i]) != len(customers):
        raise ValueError(f'Service cost data missing or incomplete for {i}.')
c = {i: {customers[j]: service_cost[i][j] for j in range(len(customers))} for i in centres}
m = gp.Model('Facility_Location')
y_vars = m.addVars(centres, vtype=GRB.BINARY, name='')
x_vars = m.addVars(centres, customers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in centres)) + gp.quicksum((c[i][j] * x_vars[i, j] for i in centres for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in centres)) == 1 for j in customers), name='')
m.addConstrs((x_vars[i, j] <= y_vars[i] for i in centres for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= 4 * y_vars[i] for i in centres), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')