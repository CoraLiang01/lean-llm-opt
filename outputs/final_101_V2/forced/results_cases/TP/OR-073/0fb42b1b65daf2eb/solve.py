import gurobipy as gp
from gurobipy import GRB
products = ['I', 'II', 'III']
equip_A = ['A1', 'A2', 'A3']
equip_B = ['B1', 'B2', 'B3', 'B4']
compat_A = {'A1': ['I', 'II'], 'A2': ['I', 'II', 'III'], 'A3': ['I', 'II', 'III']}
compat_B = {'B1': ['I', 'II'], 'B2': ['I', 'III'], 'B3': ['I'], 'B4': ['I', 'II', 'III']}
proc_time = {'A1': {'I': 5, 'II': 10}, 'A2': {'I': 7, 'II': 9, 'III': 12}, 'A3': {'I': 6, 'II': 11, 'III': 2}, 'B1': {'I': 6, 'II': 8}, 'B2': {'I': 4, 'III': 11}, 'B3': {'I': 7}, 'B4': {'I': 3, 'II': 5, 'III': 8}}
equip_time = {'A1': 6000, 'A2': 10000, 'A3': 8000, 'B1': 4000, 'B2': 7000, 'B3': 4000, 'B4': 5000}
equip_cost_full = {'A1': 300, 'A2': 321, 'A3': 203, 'B1': 250, 'B2': 783, 'B3': 200, 'B4': 300}
raw_cost = {'I': 0.25, 'II': 0.35, 'III': 0.5}
unit_price = {'I': 1.25, 'II': 2, 'III': 2.8}
m = gp.Model('Factory_Production')
x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y_A = {}
for e in equip_A:
    for p in compat_A[e]:
        y_A[e, p] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'y_{e}_{p}')
y_B = {}
for e in equip_B:
    for p in compat_B[e]:
        y_B[e, p] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'y_{e}_{p}')
t = {}
t['A1'] = gp.LinExpr()
t['A1'].addTerms([proc_time['A1']['I'], proc_time['A1']['II']], [y_A['A1', 'I'], y_A['A1', 'II']])
t['A2'] = gp.LinExpr()
t['A2'].addTerms([proc_time['A2']['I'], proc_time['A2']['II'], proc_time['A2']['III']], [y_A['A2', 'I'], y_A['A2', 'II'], y_A['A2', 'III']])
t['A3'] = gp.LinExpr()
t['A3'].addTerms([proc_time['A3']['I'], proc_time['A3']['II'], proc_time['A3']['III']], [y_A['A3', 'I'], y_A['A3', 'II'], y_A['A3', 'III']])
t['B1'] = gp.LinExpr()
t['B1'].addTerms([proc_time['B1']['I'], proc_time['B1']['II']], [y_B['B1', 'I'], y_B['B1', 'II']])
t['B2'] = gp.LinExpr()
t['B2'].addTerms([proc_time['B2']['I'], proc_time['B2']['III']], [y_B['B2', 'I'], y_B['B2', 'III']])
t['B3'] = gp.LinExpr()
t['B3'].addTerms([proc_time['B3']['I']], [y_B['B3', 'I']])
t['B4'] = gp.LinExpr()
t['B4'].addTerms([proc_time['B4']['I'], proc_time['B4']['II'], proc_time['B4']['III']], [y_B['B4', 'I'], y_B['B4', 'II'], y_B['B4', 'III']])
revenue = gp.quicksum((unit_price[p] * x[p] for p in products))
rawmat = gp.quicksum((raw_cost[p] * x[p] for p in products))
equip_cost = equip_cost_full['A1'] * t['A1'] / equip_time['A1'] + equip_cost_full['A2'] * t['A2'] / equip_time['A2'] + equip_cost_full['A3'] * t['A3'] / equip_time['A3'] + equip_cost_full['B1'] * t['B1'] / equip_time['B1'] + equip_cost_full['B2'] * t['B2'] / equip_time['B2'] + equip_cost_full['B3'] * t['B3'] / equip_time['B3'] + equip_cost_full['B4'] * t['B4'] / equip_time['B4']
m.setObjective(revenue - rawmat - equip_cost, GRB.MAXIMIZE)
m.addConstr(x['I'] == y_A['A1', 'I'] + y_A['A2', 'I'] + y_A['A3', 'I'], name='link_I_A')
m.addConstr(x['I'] == y_B['B1', 'I'] + y_B['B2', 'I'] + y_B['B3', 'I'] + y_B['B4', 'I'], name='link_I_B')
m.addConstr(x['II'] == y_A['A1', 'II'] + y_A['A2', 'II'] + y_A['A3', 'II'], name='link_II_A')
m.addConstr(x['II'] == y_B['B1', 'II'] + y_B['B4', 'II'], name='link_II_B')
m.addConstr(x['III'] == y_A['A2', 'III'] + y_A['A3', 'III'], name='link_III_A')
m.addConstr(x['III'] == y_B['B2', 'III'] + y_B['B4', 'III'], name='link_III_B')
m.addConstr(t['A1'] <= equip_time['A1'], name='cap_A1')
m.addConstr(t['A2'] <= equip_time['A2'], name='cap_A2')
m.addConstr(t['A3'] <= equip_time['A3'], name='cap_A3')
m.addConstr(t['B1'] <= equip_time['B1'], name='cap_B1')
m.addConstr(t['B2'] <= equip_time['B2'], name='cap_B2')
m.addConstr(t['B3'] <= equip_time['B3'], name='cap_B3')
m.addConstr(t['B4'] <= equip_time['B4'], name='cap_B4')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')