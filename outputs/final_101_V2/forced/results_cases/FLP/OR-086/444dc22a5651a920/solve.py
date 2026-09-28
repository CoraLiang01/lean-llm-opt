import gurobipy as gp
from gurobipy import GRB
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
supply = {'I': 1500, 'II': 2000, 'III': 1000}
cost = {'I': 6, 'II': 4.5, 'III': 3}
price = {'Red': 5.5, 'Yellow': 5, 'Blue': 4.8}
blend_reqs = {'Red': {'I': {'ub': 0.1}, 'II': {'lb': 0.5}}, 'Yellow': {'III': {'ub': 0.7}, 'I': {'lb': 0.2}}, 'Blue': {'I': {'ub': 0.5}, 'II': {'lb': 0.1}}}
min_red = 2000
for g in grades:
    if g not in supply or g not in cost:
        raise ValueError(f'Missing supply or cost data for grade {g}')
for b in brands:
    if b not in price:
        raise ValueError(f'Missing price data for brand {b}')
m = gp.Model('WineBlending')
x = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
revenue = gp.quicksum((price[b] * gp.quicksum((x[g, b] for g in grades)) for b in brands))
raw_cost = gp.quicksum((cost[g] * gp.quicksum((x[g, b] for b in brands)) for g in grades))
m.setObjective(revenue - raw_cost, GRB.MAXIMIZE)
m.addConstr(x['I', 'Red'] <= 0.1 * gp.quicksum((x[g, 'Red'] for g in grades)), name='red_I_ub')
m.addConstr(x['II', 'Red'] >= 0.5 * gp.quicksum((x[g, 'Red'] for g in grades)), name='red_II_lb')
m.addConstr(x['III', 'Yellow'] <= 0.7 * gp.quicksum((x[g, 'Yellow'] for g in grades)), name='yellow_III_ub')
m.addConstr(x['I', 'Yellow'] >= 0.2 * gp.quicksum((x[g, 'Yellow'] for g in grades)), name='yellow_I_lb')
m.addConstr(x['I', 'Blue'] <= 0.5 * gp.quicksum((x[g, 'Blue'] for g in grades)), name='blue_I_ub')
m.addConstr(x['II', 'Blue'] >= 0.1 * gp.quicksum((x[g, 'Blue'] for g in grades)), name='blue_II_lb')
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
m.addConstr(gp.quicksum((x[g, 'Red'] for g in grades)) >= min_red, name='min_red')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')