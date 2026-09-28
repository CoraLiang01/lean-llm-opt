import gurobipy as gp
from gurobipy import GRB
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
S = {'I': 1500, 'II': 2000, 'III': 1000}
c = {'I': 6, 'II': 4.5, 'III': 3}
p = {'Red': 5.5, 'Yellow': 5, 'Blue': 4.8}
blend_reqs = {('I', 'Red'): ('<', 0.1), ('II', 'Red'): ('>', 0.5), ('III', 'Yellow'): ('<', 0.7), ('I', 'Yellow'): ('>', 0.2), ('I', 'Blue'): ('<', 0.5), ('II', 'Blue'): ('>', 0.1)}
min_red = 2000
m = gp.Model('WineBlending')
x = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y = {}
for b in brands:
    y[b] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'y_{b}')
    m.addConstr(y[b] == gp.quicksum((x[g, b] for g in grades)), name=f'ydef_{b}')
m.setObjective(gp.quicksum((p[b] * y[b] for b in brands)) - gp.quicksum((c[g] * gp.quicksum((x[g, b] for b in brands)) for g in grades)), GRB.MAXIMIZE)
for (g, b), (sense, rhs) in blend_reqs.items():
    eps = 1e-06
    if sense == '<':
        m.addConstr(x[g, b] <= rhs * y[b] - eps, name=f'blend_{g}_{b}_ub')
    elif sense == '>':
        m.addConstr(x[g, b] >= rhs * y[b] + eps, name=f'blend_{g}_{b}_lb')
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= S[g], name=f'supply_{g}')
m.addConstr(y['Red'] >= min_red, name='minprod_Red')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')