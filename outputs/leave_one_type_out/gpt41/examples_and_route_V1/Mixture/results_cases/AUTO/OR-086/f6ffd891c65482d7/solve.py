LEGACY_OBSERVATION = 'Grade,Daily Supply (kg),Cost (CNY/kg)\nI,1500,6\nII,2000,4.5\nIII,1000,3\n\nBrand,Blending Requirements,Selling Price (CNY/kg)\nRed,I less than 10%  II more than 50%,5.5\nYellow,III less than 70%  I more than 20%,5\nBlue,I less than 50%  II more than 10%,4.8'
LEGACY_RECORDS = [{'source': '', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '', 'values': {'Grade': 'Brand', 'Daily Supply (kg)': 'Blending Requirements', 'Cost (CNY/kg)': 'Selling Price (CNY/kg)'}}, {'source': '', 'values': {'Grade': 'Red', 'Daily Supply (kg)': 'I less than 10%  II more than 50%', 'Cost (CNY/kg)': '5.5'}}, {'source': '', 'values': {'Grade': 'Yellow', 'Daily Supply (kg)': 'III less than 70%  I more than 20%', 'Cost (CNY/kg)': '5'}}, {'source': '', 'values': {'Grade': 'Blue', 'Daily Supply (kg)': 'I less than 50%  II more than 10%', 'Cost (CNY/kg)': '4.8'}}]
import gurobipy as gp
from gurobipy import GRB
grades = []
grade_supply = {}
grade_cost = {}
brands = []
brand_price = {}
blending_reqs = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'Grade' in v and v['Grade'] in ['I', 'II', 'III']:
        g = v['Grade']
        grades.append(g)
        grade_supply[g] = float(v['Daily Supply (kg)'])
        grade_cost[g] = float(v['Cost (CNY/kg)'])
    elif 'Grade' in v and v['Grade'] in ['Red', 'Yellow', 'Blue']:
        b = v['Grade']
        brands.append(b)
        brand_price[b] = float(v['Cost (CNY/kg)'])
        reqs = v['Daily Supply (kg)']
        blending_reqs[b] = reqs
blend_constraints = {b: [] for b in brands}
for b in brands:
    reqs = blending_reqs[b]
    tokens = reqs.replace('%', '').split()
    i = 0
    while i < len(tokens):
        if tokens[i] in grades:
            g = tokens[i]
            if tokens[i + 1] == 'less':
                op = '<'
                val = float(tokens[i + 2]) / 100
                blend_constraints[b].append((g, op, val))
                i += 3
            elif tokens[i + 1] == 'more':
                op = '>'
                val = float(tokens[i + 2]) / 100
                blend_constraints[b].append((g, op, val))
                i += 3
            else:
                raise ValueError(f'Unknown blending requirement: {tokens[i:i + 3]}')
        else:
            i += 1
m = gp.Model('wine_blend')
x = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
for b in brands:
    m.addConstr(y[b] == gp.quicksum((x[g, b] for g in grades)), name=f'ydef_{b}')
for b in brands:
    for g, op, val in blend_constraints[b]:
        if op == '<':
            m.addConstr(x[g, b] <= val * y[b] - 1e-06, name=f'blend_{b}_{g}_lt')
        elif op == '>':
            m.addConstr(x[g, b] >= val * y[b] + 1e-06, name=f'blend_{b}_{g}_gt')
        else:
            raise ValueError(f'Unknown operator {op} in blending constraints')
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
m.addConstr(y['Red'] >= 2000, name='minprod_Red')
revenue = gp.quicksum((brand_price[b] * y[b] for b in brands))
cost = gp.quicksum((grade_cost[g] * gp.quicksum((x[g, b] for b in brands)) for g in grades))
m.setObjective(revenue - cost, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')