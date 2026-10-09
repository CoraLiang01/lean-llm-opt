import gurobipy as gp
from gurobipy import GRB
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
supply_limits = {'I': 1500, 'II': 2000, 'III': 1000}
unit_costs = {'I': 6, 'II': 4.5, 'III': 3}
selling_prices = {'Red': 5.5, 'Yellow': 5, 'Blue': 4.8}
blending_bounds = {('I', 'Red'): (None, 0.1), ('II', 'Red'): (0.5, None), ('III', 'Yellow'): (None, 0.7), ('I', 'Yellow'): (0.2, None), ('I', 'Blue'): (None, 0.5), ('II', 'Blue'): (0.1, None)}
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
for b in brands:
    m.addConstr(y_vars[b] == gp.quicksum((x_vars[g, b] for g in grades)), name=f'ydef_{b}')
m.addConstr(x_vars['I', 'Red'] <= 0.1 * y_vars['Red'], name='blend_I_Red_ub')
m.addConstr(x_vars['II', 'Red'] >= 0.5 * y_vars['Red'], name='blend_II_Red_lb')
m.addConstr(x_vars['III', 'Yellow'] <= 0.7 * y_vars['Yellow'], name='blend_III_Yellow_ub')
m.addConstr(x_vars['I', 'Yellow'] >= 0.2 * y_vars['Yellow'], name='blend_I_Yellow_lb')
m.addConstr(x_vars['I', 'Blue'] <= 0.5 * y_vars['Blue'], name='blend_I_Blue_ub')
m.addConstr(x_vars['II', 'Blue'] >= 0.1 * y_vars['Blue'], name='blend_II_Blue_lb')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply_limits[g], name=f'supply_{g}')
m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
revenue = gp.quicksum((selling_prices[b] * y_vars[b] for b in brands))
cost = gp.quicksum((unit_costs[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(revenue - cost, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')