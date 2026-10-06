LEGACY_OBSERVATION = '30-1.csv\nGrade,Daily Supply (kg),Cost (CNY/kg)\nI,1500,6\nII,2000,4.5\nIII,1000,3\n\n30-2.csv\nBrand,Blending Requirements,Selling Price (CNY/kg)\nRed,I less than 10%  II more than 50%,5.5\nYellow,III less than 70%  I more than 20%,5\nBlue,I less than 50%  II more than 10%,4.8'
LEGACY_RECORDS = [{'source': '30-1.csv', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '30-1.csv', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '30-1.csv', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '30-2.csv', 'values': {'Brand': 'Red', 'Blending Requirements': 'I less than 10%  II more than 50%', 'Selling Price (CNY/kg)': '5.5'}}, {'source': '30-2.csv', 'values': {'Brand': 'Yellow', 'Blending Requirements': 'III less than 70%  I more than 20%', 'Selling Price (CNY/kg)': '5'}}, {'source': '30-2.csv', 'values': {'Brand': 'Blue', 'Blending Requirements': 'I less than 50%  II more than 10%', 'Selling Price (CNY/kg)': '4.8'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '30-1.csv', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '30-1.csv', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '30-1.csv', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '30-2.csv', 'values': {'Brand': 'Red', 'Blending Requirements': 'I less than 10%  II more than 50%', 'Selling Price (CNY/kg)': '5.5'}}, {'source': '30-2.csv', 'values': {'Brand': 'Yellow', 'Blending Requirements': 'III less than 70%  I more than 20%', 'Selling Price (CNY/kg)': '5'}}, {'source': '30-2.csv', 'values': {'Brand': 'Blue', 'Blending Requirements': 'I less than 50%  II more than 10%', 'Selling Price (CNY/kg)': '4.8'}}]
grades = []
grade_supply = {}
grade_cost = {}
brands = []
brand_price = {}
blend_reqs = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == '30-1.csv':
        g = rec['values']['Grade']
        grades.append(g)
        grade_supply[g] = float(rec['values']['Daily Supply (kg)'])
        grade_cost[g] = float(rec['values']['Cost (CNY/kg)'])
    elif rec['source'] == '30-2.csv':
        b = rec['values']['Brand']
        brands.append(b)
        brand_price[b] = float(rec['values']['Selling Price (CNY/kg)'])
        blend_reqs[b] = rec['values']['Blending Requirements']
grades = list(dict.fromkeys(grades))
brands = list(dict.fromkeys(brands))
blend_constraints = []
for b in brands:
    req = blend_reqs[b]
    tokens = req.split()
    i = 0
    while i < len(tokens):
        if tokens[i] in grades:
            g = tokens[i]
            if tokens[i + 1] == 'less':
                bound = float(tokens[i + 3].strip('%')) / 100
                blend_constraints.append((g, b, '<=', bound))
                i += 4
            elif tokens[i + 1] == 'more':
                bound = float(tokens[i + 3].strip('%')) / 100
                blend_constraints.append((g, b, '>=', bound))
                i += 4
            else:
                raise ValueError(f'Unknown blending requirement: {tokens[i:i + 4]}')
        else:
            i += 1
m = gp.Model('wine_blend')
x = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
for b in brands:
    m.addConstr(y[b] == gp.quicksum((x[g, b] for g in grades)), name=f'ydef_{b}')
m.setObjective(gp.quicksum((brand_price[b] * y[b] for b in brands)) - gp.quicksum((grade_cost[g] * gp.quicksum((x[g, b] for b in brands)) for g in grades)), GRB.MAXIMIZE)
for g in grades:
    m.addConstr(gp.quicksum((x[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
m.addConstr(y['Red'] >= 2000, name='minprod_Red')
for g, b, sense, bound in blend_constraints:
    if sense == '<=':
        m.addConstr(x[g, b] <= bound * y[b], name=f'blend_{g}_{b}_le')
    elif sense == '>=':
        m.addConstr(x[g, b] >= bound * y[b], name=f'blend_{g}_{b}_ge')
    else:
        raise ValueError(f'Unknown sense {sense} in blending constraints')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')