import gurobipy as gp
from gurobipy import GRB
equipments = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4']
products = ['I', 'II', 'III']
product_idx = {'I': 1, 'II': 2, 'III': 3}
processing_time = {('A1', 'I'): 5, ('A1', 'II'): 10, ('A2', 'I'): 7, ('A2', 'II'): 9, ('A2', 'III'): 12, ('A3', 'I'): 6, ('A3', 'II'): 11, ('A3', 'III'): 2, ('B1', 'I'): 6, ('B1', 'II'): 8, ('B2', 'I'): 4, ('B2', 'III'): 11, ('B3', 'I'): 7, ('B4', 'I'): 3, ('B4', 'II'): 5, ('B4', 'III'): 8}
available_time = {'A1': 6000, 'A2': 10000, 'A3': 8000, 'B1': 4000, 'B2': 7000, 'B3': 4000, 'B4': 5000}
equipment_cost = {'A1': 300, 'A2': 321, 'A3': 203, 'B1': 250, 'B2': 783, 'B3': 200, 'B4': 300}
raw_material_cost = {'I': 0.25, 'II': 0.35, 'III': 0.5}
unit_price = {'I': 1.25, 'II': 2, 'III': 2.8}
eligible_pairs = list(processing_time.keys())
m = gp.Model('Factory_Production_Optimization')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(eligible_pairs, lb=0, vtype=GRB.CONTINUOUS, name='')
profit_expr = (unit_price['I'] - raw_material_cost['I']) * x_vars['I'] + (unit_price['II'] - raw_material_cost['II']) * x_vars['II'] + (unit_price['III'] - raw_material_cost['III']) * x_vars['III']
equipment_cost_expr = gp.LinExpr()
for k in equipments:
    eligible_products = [i for (kk, i) in eligible_pairs if kk == k]
    if eligible_products:
        total_time = gp.quicksum((processing_time[k, i] * y_vars[k, i] for i in eligible_products))
        equipment_cost_expr += equipment_cost[k] * total_time / available_time[k]
m.setObjective(profit_expr - equipment_cost_expr, GRB.MAXIMIZE)
m.addConstr(x_vars['I'] == y_vars['A1', 'I'] + y_vars['A2', 'I'] + y_vars['A3', 'I'], name='A_I_assign')
m.addConstr(x_vars['II'] == y_vars['A1', 'II'] + y_vars['A2', 'II'] + y_vars['A3', 'II'], name='A_II_assign')
m.addConstr(x_vars['III'] == y_vars['A2', 'III'] + y_vars['A3', 'III'], name='A_III_assign')
m.addConstr(x_vars['I'] == y_vars['B1', 'I'] + y_vars['B2', 'I'] + y_vars['B3', 'I'] + y_vars['B4', 'I'], name='B_I_assign')
m.addConstr(x_vars['II'] == y_vars['B1', 'II'] + y_vars['B4', 'II'], name='B_II_assign')
m.addConstr(x_vars['III'] == y_vars['B2', 'III'] + y_vars['B4', 'III'], name='B_III_assign')
for k in equipments:
    eligible_products = [i for (kk, i) in eligible_pairs if kk == k]
    if eligible_products:
        m.addConstr(gp.quicksum((processing_time[k, i] * y_vars[k, i] for i in eligible_products)) <= available_time[k], name=f'{k}_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')