import gurobipy as gp
from gurobipy import GRB
raw_grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
raw_materials = {'I': {'daily_supply': 1500, 'cost': 6}, 'II': {'daily_supply': 2000, 'cost': 4.5}, 'III': {'daily_supply': 1000, 'cost': 3}}
brand_data = {'Red': {'blending_requirements': {'I': {'upper': 0.1}, 'II': {'lower': 0.5}}, 'selling_price': 5.5}, 'Yellow': {'blending_requirements': {'III': {'upper': 0.7}, 'I': {'lower': 0.2}}, 'selling_price': 5}, 'Blue': {'blending_requirements': {'I': {'upper': 0.5}, 'II': {'lower': 0.1}}, 'selling_price': 4.8}}
minimum_production = {'Red': 2000}
for i in raw_grades:
    if i not in raw_materials or 'daily_supply' not in raw_materials[i] or 'cost' not in raw_materials[i]:
        raise ValueError(f'Missing data for raw grade {i}')
for j in brands:
    if j not in brand_data or 'selling_price' not in brand_data[j] or 'blending_requirements' not in brand_data[j]:
        raise ValueError(f'Missing data for brand {j}')
m = gp.Model('WineBlending')
x = m.addVars(raw_grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
for j in brands:
    m.addConstr(y[j] == gp.quicksum((x[i, j] for i in raw_grades)), name=f'ydef_{j}')
m.setObjective(gp.quicksum((brand_data[j]['selling_price'] * y[j] for j in brands)) - gp.quicksum((raw_materials[i]['cost'] * gp.quicksum((x[i, j] for j in brands)) for i in raw_grades)), GRB.MAXIMIZE)
m.addConstr(x['I', 'Red'] <= 0.1 * y['Red'] - 1e-06, name='blend_Red_I_upper')
m.addConstr(x['II', 'Red'] >= 0.5 * y['Red'] + 1e-06, name='blend_Red_II_lower')
m.addConstr(x['III', 'Yellow'] <= 0.7 * y['Yellow'] - 1e-06, name='blend_Yellow_III_upper')
m.addConstr(x['I', 'Yellow'] >= 0.2 * y['Yellow'] + 1e-06, name='blend_Yellow_I_lower')
m.addConstr(x['I', 'Blue'] <= 0.5 * y['Blue'] - 1e-06, name='blend_Blue_I_upper')
m.addConstr(x['II', 'Blue'] >= 0.1 * y['Blue'] + 1e-06, name='blend_Blue_II_lower')
for i in raw_grades:
    m.addConstr(gp.quicksum((x[i, j] for j in brands)) <= raw_materials[i]['daily_supply'], name=f'supply_{i}')
m.addConstr(y['Red'] >= minimum_production['Red'], name='minprod_Red')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')