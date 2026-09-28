import gurobipy as gp
from gurobipy import GRB
products = ['Product I', 'Product II', 'Product III']
equipments = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4']
processing_time = {'A1': {'Product I': 5, 'Product II': 10}, 'A2': {'Product I': 7, 'Product II': 9, 'Product III': 12}, 'A3': {'Product I': 6, 'Product II': 11, 'Product III': 2}, 'B1': {'Product I': 6, 'Product II': 8}, 'B2': {'Product I': 4, 'Product III': 11}, 'B3': {'Product I': 7}, 'B4': {'Product I': 3, 'Product II': 5, 'Product III': 8}}
available_time = {'A1': 6000, 'A2': 10000, 'A3': 8000, 'B1': 4000, 'B2': 7000, 'B3': 4000, 'B4': 5000}
equipment_cost_full_load = {'A1': 300, 'A2': 321, 'A3': 203, 'B1': 250, 'B2': 783, 'B3': 200, 'B4': 300}
raw_material_cost = {'Product I': 0.25, 'Product II': 0.35, 'Product III': 0.5}
unit_price = {'Product I': 1.25, 'Product II': 2, 'Product III': 2.8}
for e in equipments:
    if e not in processing_time or e not in available_time or e not in equipment_cost_full_load:
        raise ValueError(f'Missing data for equipment {e}')
for p in products:
    if p not in raw_material_cost or p not in unit_price:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('Factory_Production')
x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(equipments, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='')
revenue = gp.quicksum((unit_price[p] * x[p] for p in products))
raw_cost = gp.quicksum((raw_material_cost[p] * x[p] for p in products))
equip_cost = gp.quicksum((equipment_cost_full_load[e] * y[e] for e in equipments))
m.setObjective(revenue - raw_cost - equip_cost, GRB.MAXIMIZE)
m.addConstr(5 * x['Product I'] + 10 * x['Product II'] <= 6000 * y['A1'], name='A1_time')
m.addConstr(7 * x['Product I'] + 9 * x['Product II'] + 12 * x['Product III'] <= 10000 * y['A2'], name='A2_time')
m.addConstr(6 * x['Product I'] + 11 * x['Product II'] + 2 * x['Product III'] <= 8000 * y['A3'], name='A3_time')
m.addConstr(6 * x['Product I'] + 8 * x['Product II'] <= 4000 * y['B1'], name='B1_time')
m.addConstr(4 * x['Product I'] + 11 * x['Product III'] <= 7000 * y['B2'], name='B2_time')
m.addConstr(7 * x['Product I'] <= 4000 * y['B3'], name='B3_time')
m.addConstr(3 * x['Product I'] + 5 * x['Product II'] + 8 * x['Product III'] <= 5000 * y['B4'], name='B4_time')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')