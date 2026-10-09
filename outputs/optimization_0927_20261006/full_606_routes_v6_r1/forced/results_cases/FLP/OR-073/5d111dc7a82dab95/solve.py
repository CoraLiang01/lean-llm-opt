import gurobipy as gp
from gurobipy import GRB
products = ['I', 'II', 'III']
A_equip = ['A1', 'A2']
B_equip = ['B1', 'B2', 'B3']
processing_time = {'A1': {'I': 5, 'II': 10}, 'A2': {'I': 7, 'II': 9, 'III': 12}, 'B1': {'I': 6, 'II': 8}, 'B2': {'I': 4, 'III': 11}, 'B3': {'I': 7}}
available_time = {'A1': 6000, 'A2': 10000, 'B1': 4000, 'B2': 7000, 'B3': 4000}
equipment_cost_full_load = {'A1': 300, 'A2': 321, 'B1': 250, 'B2': 783, 'B3': 200}
raw_material_cost = {'I': 0.25, 'II': 0.35, 'III': 0.5}
unit_price = {'I': 1.25, 'II': 2, 'III': 2.8}
A_vars_list = [('A1', 'I'), ('A1', 'II'), ('A2', 'I'), ('A2', 'II'), ('A2', 'III')]
B_vars_list = [('B1', 'I'), ('B1', 'II'), ('B2', 'I'), ('B2', 'III'), ('B3', 'I')]
for (k, p) in A_vars_list:
    if k not in processing_time or p not in processing_time[k]:
        raise ValueError(f'Missing processing_time for {k},{p}')
    if k not in available_time:
        raise ValueError(f'Missing available_time for {k}')
    if k not in equipment_cost_full_load:
        raise ValueError(f'Missing equipment_cost_full_load for {k}')
for (l, p) in B_vars_list:
    if l not in processing_time or p not in processing_time[l]:
        raise ValueError(f'Missing processing_time for {l},{p}')
    if l not in available_time:
        raise ValueError(f'Missing available_time for {l}')
    if l not in equipment_cost_full_load:
        raise ValueError(f'Missing equipment_cost_full_load for {l}')
for p in products:
    if p not in raw_material_cost:
        raise ValueError(f'Missing raw_material_cost for {p}')
    if p not in unit_price:
        raise ValueError(f'Missing unit_price for {p}')
m = gp.Model('factory_production')
A_vars = m.addVars(A_vars_list, lb=0, vtype=GRB.CONTINUOUS, name='')
B_vars = m.addVars(B_vars_list, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
profit_expr = gp.quicksum((unit_price[p] * y_vars[p] for p in products)) - gp.quicksum((raw_material_cost[p] * y_vars[p] for p in products)) - (equipment_cost_full_load['A1'] / available_time['A1'] * (processing_time['A1']['I'] * A_vars['A1', 'I'] + processing_time['A1']['II'] * A_vars['A1', 'II']) + equipment_cost_full_load['A2'] / available_time['A2'] * (processing_time['A2']['I'] * A_vars['A2', 'I'] + processing_time['A2']['II'] * A_vars['A2', 'II'] + processing_time['A2']['III'] * A_vars['A2', 'III'])) - (equipment_cost_full_load['B1'] / available_time['B1'] * (processing_time['B1']['I'] * B_vars['B1', 'I'] + processing_time['B1']['II'] * B_vars['B1', 'II']) + equipment_cost_full_load['B2'] / available_time['B2'] * (processing_time['B2']['I'] * B_vars['B2', 'I'] + processing_time['B2']['III'] * B_vars['B2', 'III']) + equipment_cost_full_load['B3'] / available_time['B3'] * (processing_time['B3']['I'] * B_vars['B3', 'I']))
m.setObjective(profit_expr, GRB.MAXIMIZE)
m.addConstr(A_vars['A1', 'I'] + A_vars['A2', 'I'] == y_vars['I'], name='balA_I')
m.addConstr(A_vars['A1', 'II'] + A_vars['A2', 'II'] == y_vars['II'], name='balA_II')
m.addConstr(A_vars['A2', 'III'] == y_vars['III'], name='balA_III')
m.addConstr(B_vars['B1', 'I'] + B_vars['B2', 'I'] + B_vars['B3', 'I'] == y_vars['I'], name='balB_I')
m.addConstr(B_vars['B1', 'II'] == y_vars['II'], name='balB_II')
m.addConstr(B_vars['B2', 'III'] == y_vars['III'], name='balB_III')
m.addConstr(processing_time['A1']['I'] * A_vars['A1', 'I'] + processing_time['A1']['II'] * A_vars['A1', 'II'] <= available_time['A1'], name='time_A1')
m.addConstr(processing_time['A2']['I'] * A_vars['A2', 'I'] + processing_time['A2']['II'] * A_vars['A2', 'II'] + processing_time['A2']['III'] * A_vars['A2', 'III'] <= available_time['A2'], name='time_A2')
m.addConstr(processing_time['B1']['I'] * B_vars['B1', 'I'] + processing_time['B1']['II'] * B_vars['B1', 'II'] <= available_time['B1'], name='time_B1')
m.addConstr(processing_time['B2']['I'] * B_vars['B2', 'I'] + processing_time['B2']['III'] * B_vars['B2', 'III'] <= available_time['B2'], name='time_B2')
m.addConstr(processing_time['B3']['I'] * B_vars['B3', 'I'] <= available_time['B3'], name='time_B3')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')