LEGACY_OBSERVATION = 'truck_id,Q,S,C,d1,d2,d3,d4\n1,1000,500,2.0,1500,2000,1800,1000\n2,800,300,3.0,1500,2000,1800,1000\n3,1200,400,2.5,1500,2000,1800,1000\n4,600,250,3.0,1500,2000,1800,1000\n5,900,450,2.2,1500,2000,1800,1000\n6,700,280,2.8,1500,2000,1800,1000\n7,1100,420,2.4,1500,2000,1800,1000\n8,500,200,3.2,1500,2000,1800,1000\n9,1000,480,2.1,1500,2000,1800,1000\n10,650,260,2.9,1500,2000,1800,1000'
LEGACY_RECORDS = [{'source': '', 'values': {'truck_id': '1', 'Q': '1000', 'S': '500', 'C': '2.0', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000'}}, {'source': '', 'values': {'truck_id': '2', 'Q': '800', 'S': '300', 'C': '3.0', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000'}}, {'source': '', 'values': {'truck_id': '3', 'Q': '1200', 'S': '400', 'C': '2.5', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000'}}, {'source': '', 'values': {'truck_id': '4', 'Q': '600', 'S': '250', 'C': '3.0', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000'}}, {'source': '', 'values': {'truck_id': '5', 'Q': '900', 'S': '450', 'C': '2.2', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000'}}, {'source': '', 'values': {'truck_id': '6', 'Q': '700', 'S': '280', 'C': '2.8', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000'}}, {'source': '', 'values': {'truck_id': '7', 'Q': '1100', 'S': '420', 'C': '2.4', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000'}}, {'source': '', 'values': {'truck_id': '8', 'Q': '500', 'S': '200', 'C': '3.2', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000'}}, {'source': '', 'values': {'truck_id': '9', 'Q': '1000', 'S': '480', 'C': '2.1', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000'}}, {'source': '', 'values': {'truck_id': '10', 'Q': '650', 'S': '260', 'C': '2.9', 'd1': '1500', 'd2': '2000', 'd3': '1800', 'd4': '1000'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
truck_ids = []
Q = {}
S = {}
C = {}
for rec in records:
    v = rec['values']
    tid = str(v['truck_id'])
    truck_ids.append(tid)
    Q[tid] = float(v['Q'])
    S[tid] = float(v['S'])
    C[tid] = float(v['C'])
T = ['1', '2', '3', '4']
T_int = [1, 2, 3, 4]
demand = {}
for t in T:
    key = f'd{t}'
    demand[t] = float(records[0]['values'][key])
m = gp.Model('Truck_Scheduling')
y = m.addVars(truck_ids, T, vtype=GRB.BINARY, name='')
u = m.addVars(truck_ids, T, vtype=GRB.BINARY, name='')
x = m.addVars(truck_ids, T, lb=0, vtype=GRB.CONTINUOUS, name='')
y0 = {i: 0 for i in truck_ids}
x0 = {i: 0 for i in truck_ids}
m.setObjective(gp.quicksum((S[i] * u[i, t] + C[i] * x[i, t] for i in truck_ids for t in T)), GRB.MINIMIZE)
for i in truck_ids:
    m.addConstr(u[i, '1'] >= y[i, '1'], name=f'startup1_{i}')
for i in truck_ids:
    for t in ['2', '3', '4']:
        m.addConstr(u[i, t] >= y[i, t] - (y[i, str(int(t) - 1)] if int(t) - 1 >= 1 else y0[i]), name=f'startup2_{i}_{t}')
for i in truck_ids:
    for t in T:
        prev = y[i, str(int(t) - 1)] if int(t) - 1 >= 1 else y0[i]
        m.addConstr(u[i, t] <= 1 - prev, name=f'startup3_{i}_{t}')
for i in truck_ids:
    m.addConstr(u[i, '4'] == 0, name=f'no_startup4_{i}')
for i in truck_ids:
    for t in ['1', '2', '3']:
        next_t = str(int(t) + 1)
        m.addConstr(y[i, next_t] >= y[i, t] - u[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    for t in ['2', '3']:
        prev = y[i, str(int(t) - 1)] if int(t) - 1 >= 1 else y0[i]
        curr = y[i, t]
        next_t = y[i, str(int(t) + 1)] if int(t) + 1 <= 4 else 0
        m.addConstr(prev - curr <= 1 - next_t, name=f'min_down_{i}_{t}')
for i in truck_ids:
    for t in T:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x[i, t] >= 0, name=f'nonneg_x_{i}_{t}')
for i in truck_ids:
    for t in ['2', '3', '4']:
        prev = x[i, str(int(t) - 1)] if int(t) - 1 >= 1 else x0[i]
        m.addConstr(x[i, t] - prev <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x[i, t] - prev >= -300, name=f'ramp_down_{i}_{t}')
for t in T:
    m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in T:
    m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in truck_ids)), name=f'spare_cap_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')