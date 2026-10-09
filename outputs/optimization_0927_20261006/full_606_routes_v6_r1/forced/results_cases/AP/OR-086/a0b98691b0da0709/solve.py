import gurobipy as gp
from gurobipy import GRB
raw_grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
cost = {'I': 6, 'II': 4.5, 'III': 3}
supply = {'I': 1500, 'II': 2000, 'III': 1000}
price = {'Red': 5.5, 'Yellow': 5, 'Blue': 4.8}
blending_reqs = {('I', 'Red'): ('upper', 0.1), ('II', 'Red'): ('lower', 0.5), ('III', 'Yellow'): ('upper', 0.7), ('I', 'Yellow'): ('lower', 0.2), ('I', 'Blue'): ('upper', 0.5), ('II', 'Blue'): ('lower', 0.1)}
m = gp.Model('WineBlending')
x_vars = m.addVars(raw_grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((price[j] * y_vars[j] for j in brands)) - gp.quicksum((cost[i] * gp.quicksum((x_vars[i, j] for j in brands)) for i in raw_grades)), GRB.MAXIMIZE)
for j in brands:
    m.addConstr(y_vars[j] == gp.quicksum((x_vars[i, j] for i in raw_grades)), name=f'ydef_{j}')
for ((i, j), (typ, bound)) in blending_reqs.items():
    if typ == 'upper':
        m.addConstr(x_vars[i, j] <= bound * y_vars[j], name=f'blendU_{i}_{j}')
    elif typ == 'lower':
        m.addConstr(x_vars[i, j] >= bound * y_vars[j], name=f'blendL_{i}_{j}')
for i in raw_grades:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in brands)) <= supply[i], name=f'supply_{i}')
m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')