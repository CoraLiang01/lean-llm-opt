import gurobipy as gp
from gurobipy import GRB
products = ['I', 'II', 'III']
A_equip = ['A1', 'A2']
B_equip = ['B1', 'B2', 'B3']
equip = A_equip + B_equip
A_eligible = {('I', 'A1'), ('I', 'A2'), ('II', 'A1'), ('II', 'A2'), ('III', 'A2')}
B_eligible = {('I', 'B1'), ('I', 'B2'), ('I', 'B3'), ('II', 'B1'), ('III', 'B2')}
processing_time = {'A1': {'I': 5, 'II': 10}, 'A2': {'I': 7, 'II': 9, 'III': 12}, 'B1': {'I': 6, 'II': 8}, 'B2': {'I': 4, 'III': 11}, 'B3': {'I': 7}}
available_time = {'A1': 6000, 'A2': 10000, 'B1': 4000, 'B2': 7000, 'B3': 4000}
equipment_cost = {'A1': 300, 'A2': 321, 'B1': 250, 'B2': 783, 'B3': 200}
raw_material_cost = {'I': 0.25, 'II': 0.35, 'III': 0.5}
unit_price = {'I': 1.25, 'II': 2, 'III': 2.8}
for e in A_equip:
    for p in products:
        if (p, e) in A_eligible:
            if p not in processing_time[e]:
                raise ValueError(f'Missing processing_time for {e}, {p}')
for e in B_equip:
    for p in products:
        if (p, e) in B_eligible:
            if p not in processing_time[e]:
                raise ValueError(f'Missing processing_time for {e}, {p}')
for e in equip:
    if e not in available_time or e not in equipment_cost:
        raise ValueError(f'Missing available_time or equipment_cost for {e}')
for p in products:
    if p not in raw_material_cost or p not in unit_price:
        raise ValueError(f'Missing raw_material_cost or unit_price for {p}')
m = gp.Model('Factory_Production')
A_vars = m.addVars(((p, e) for (p, e) in A_eligible), lb=0, vtype=GRB.CONTINUOUS, name='')
B_vars = m.addVars(((p, e) for (p, e) in B_eligible), lb=0, vtype=GRB.CONTINUOUS, name='')
x_prod_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.addConstr(x_prod_vars['I'] == A_vars['I', 'A1'] + A_vars['I', 'A2'], name='I_A_sum')
m.addConstr(x_prod_vars['I'] == B_vars['I', 'B1'] + B_vars['I', 'B2'] + B_vars['I', 'B3'], name='I_B_sum')
m.addConstr(x_prod_vars['II'] == A_vars['II', 'A1'] + A_vars['II', 'A2'], name='II_A_sum')
m.addConstr(x_prod_vars['II'] == B_vars['II', 'B1'], name='II_B_sum')
m.addConstr(x_prod_vars['III'] == A_vars['III', 'A2'], name='III_A_sum')
m.addConstr(x_prod_vars['III'] == B_vars['III', 'B2'], name='III_B_sum')
m.addConstr(5 * A_vars['I', 'A1'] + 10 * A_vars['II', 'A1'] <= available_time['A1'], name='A1_time')
m.addConstr(7 * A_vars['I', 'A2'] + 9 * A_vars['II', 'A2'] + 12 * A_vars['III', 'A2'] <= available_time['A2'], name='A2_time')
m.addConstr(6 * B_vars['I', 'B1'] + 8 * B_vars['II', 'B1'] <= available_time['B1'], name='B1_time')
m.addConstr(4 * B_vars['I', 'B2'] + 11 * B_vars['III', 'B2'] <= available_time['B2'], name='B2_time')
m.addConstr(7 * B_vars['I', 'B3'] <= available_time['B3'], name='B3_time')
profit_expr = (unit_price['I'] - raw_material_cost['I']) * x_prod_vars['I'] + (unit_price['II'] - raw_material_cost['II']) * x_prod_vars['II'] + (unit_price['III'] - raw_material_cost['III']) * x_prod_vars['III'] - (equipment_cost['A1'] + equipment_cost['A2'] + equipment_cost['B1'] + equipment_cost['B2'] + equipment_cost['B3'])
m.setObjective(profit_expr, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')