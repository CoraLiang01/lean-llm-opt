import gurobipy as gp
from gurobipy import GRB
grades = ['I', 'II', 'III']
brands = ['Red', 'Yellow', 'Blue']
cost = {'I': 6, 'II': 4.5, 'III': 3}
supply = {'I': 1500, 'II': 2000, 'III': 1000}
price = {'Red': 5.5, 'Yellow': 5, 'Blue': 4.8}
blend_req = {('I', 'Red'): (None, 0.1), ('II', 'Red'): (0.5, None), ('III', 'Yellow'): (None, 0.7), ('I', 'Yellow'): (0.2, None), ('I', 'Blue'): (None, 0.5), ('II', 'Blue'): (0.1, None)}
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((price[b] * gp.quicksum((x_vars[g, b] for g in grades)) for b in brands)) - gp.quicksum((cost[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades)), GRB.MAXIMIZE)
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= supply[g], name=f'supply_{g}')
m.addConstr(gp.quicksum((x_vars[g, 'Red'] for g in grades)) >= 2000, name='minprod_Red')
bigM = 1000000.0
epsilon = 0.0001
total_prod = {}
for b in brands:
    total_prod[b] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'total_{b}')
    m.addConstr(total_prod[b] == gp.quicksum((x_vars[g, b] for g in grades)), name=f'totaldef_{b}')
for ((g, b), (lower, upper)) in blend_req.items():
    is_prod_b = m.addVar(vtype=GRB.BINARY, name=f'isprod_{b}')
    m.addConstr(total_prod[b] >= epsilon * is_prod_b, name=f'prodflag_lb_{b}')
    m.addConstr(total_prod[b] <= bigM * is_prod_b, name=f'prodflag_ub_{b}')
    if upper is not None:
        m.addGenConstrIndicator(is_prod_b, True, x_vars[g, b] <= (upper - epsilon) * total_prod[b], name=f'blend_ub_{g}_{b}')
    if lower is not None:
        m.addGenConstrIndicator(is_prod_b, True, x_vars[g, b] >= (lower + epsilon) * total_prod[b], name=f'blend_lb_{g}_{b}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')