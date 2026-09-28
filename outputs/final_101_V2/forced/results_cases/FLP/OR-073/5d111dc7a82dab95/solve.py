import gurobipy as gp
from gurobipy import GRB
products = ['I', 'II', 'III']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
t = {('A1', 'I'): 5, ('A1', 'II'): 10, ('A2', 'I'): 7, ('A2', 'II'): 9, ('A2', 'III'): 12, ('B1', 'I'): 6, ('B1', 'II'): 8, ('B2', 'I'): 4, ('B2', 'III'): 11, ('B3', 'I'): 7}
T = {'A1': 6000, 'A2': 10000, 'B1': 4000, 'B2': 7000, 'B3': 4000}
C = {'A1': 300, 'A2': 321, 'B1': 250, 'B2': 783, 'B3': 200}
r = {'I': 0.25, 'II': 0.35, 'III': 0.5}
s = {'I': 1.25, 'II': 2, 'III': 2.8}
m = gp.Model('Factory_Production')
x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y_index = []
for e in equip_A + equip_B:
    for p in products:
        if (e, p) in t:
            y_index.append((e, p))
y = m.addVars(y_index, lb=0, vtype=GRB.CONTINUOUS, name='')
obj_expr = gp.quicksum((s[p] * x[p] for p in products)) - gp.quicksum((r[p] * x[p] for p in products)) - (C['A1'] / T['A1'] * (t['A1', 'I'] * y['A1', 'I'] + t['A1', 'II'] * y['A1', 'II']) + C['A2'] / T['A2'] * (t['A2', 'I'] * y['A2', 'I'] + t['A2', 'II'] * y['A2', 'II'] + t['A2', 'III'] * y['A2', 'III']) + C['B1'] / T['B1'] * (t['B1', 'I'] * y['B1', 'I'] + t['B1', 'II'] * y['B1', 'II']) + C['B2'] / T['B2'] * (t['B2', 'I'] * y['B2', 'I'] + t['B2', 'III'] * y['B2', 'III']) + C['B3'] / T['B3'] * (t['B3', 'I'] * y['B3', 'I']))
m.setObjective(obj_expr, GRB.MAXIMIZE)
m.addConstr(x['I'] == y['A1', 'I'] + y['A2', 'I'], name='flow_I_A')
m.addConstr(x['I'] == y['B1', 'I'] + y['B2', 'I'] + y['B3', 'I'], name='flow_I_B')
m.addConstr(x['II'] == y['A1', 'II'] + y['A2', 'II'], name='flow_II_A')
m.addConstr(x['II'] == y['B1', 'II'], name='flow_II_B')
m.addConstr(x['III'] == y['A2', 'III'], name='flow_III_A')
m.addConstr(x['III'] == y['B2', 'III'], name='flow_III_B')
m.addConstr(t['A1', 'I'] * y['A1', 'I'] + t['A1', 'II'] * y['A1', 'II'] <= T['A1'], name='cap_A1')
m.addConstr(t['A2', 'I'] * y['A2', 'I'] + t['A2', 'II'] * y['A2', 'II'] + t['A2', 'III'] * y['A2', 'III'] <= T['A2'], name='cap_A2')
m.addConstr(t['B1', 'I'] * y['B1', 'I'] + t['B1', 'II'] * y['B1', 'II'] <= T['B1'], name='cap_B1')
m.addConstr(t['B2', 'I'] * y['B2', 'I'] + t['B2', 'III'] * y['B2', 'III'] <= T['B2'], name='cap_B2')
m.addConstr(t['B3', 'I'] * y['B3', 'I'] <= T['B3'], name='cap_B3')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')