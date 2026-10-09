LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "I",\n      "Daily Supply (kg)": "1500",\n      "Cost (CNY/kg)": "6"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "II",\n      "Daily Supply (kg)": "2000",\n      "Cost (CNY/kg)": "4.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "III",\n      "Daily Supply (kg)": "1000",\n      "Cost (CNY/kg)": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Red",\n      "Blending Requirements": "I less than 10%  II more than 50%",\n      "Selling Price (CNY/kg)": "5.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Yellow",\n      "Blending Requirements": "III less than 70%  I more than 20%",\n      "Selling Price (CNY/kg)": "5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Blue",\n      "Blending Requirements": "I less than 50%  II more than 10%",\n      "Selling Price (CNY/kg)": "4.8"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Red', 'Blending Requirements': 'I less than 10%  II more than 50%', 'Selling Price (CNY/kg)': '5.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Yellow', 'Blending Requirements': 'III less than 70%  I more than 20%', 'Selling Price (CNY/kg)': '5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Blue', 'Blending Requirements': 'I less than 50%  II more than 10%', 'Selling Price (CNY/kg)': '4.8'}}]
import gurobipy as gp
from gurobipy import GRB
legacy = LEGACY_RECORDS
grades = []
grade_supply = {}
grade_cost = {}
for rec in legacy:
    if rec['source'].endswith('30-1.csv'):
        g = rec['values']['Grade']
        grades.append(g)
        grade_supply[g] = float(rec['values']['Daily Supply (kg)'])
        grade_cost[g] = float(rec['values']['Cost (CNY/kg)'])
brands = []
brand_price = {}
brand_blend = {}
for rec in legacy:
    if rec['source'].endswith('30-2.csv'):
        b = rec['values']['Brand']
        brands.append(b)
        brand_price[b] = float(rec['values']['Selling Price (CNY/kg)'])
        brand_blend[b] = rec['values']['Blending Requirements']
blend_ub = {}
blend_lb = {}
for b in brands:
    req = brand_blend[b]
    tokens = req.replace('%', '').split()
    i = 0
    while i < len(tokens):
        if tokens[i] in grades:
            g = tokens[i]
            if tokens[i + 1] == 'less':
                ub = float(tokens[i + 3]) / 100.0
                blend_ub[g, b] = ub
                i += 4
            elif tokens[i + 1] == 'more':
                lb = float(tokens[i + 3]) / 100.0
                blend_lb[g, b] = lb
                i += 4
            else:
                raise ValueError(f'Unknown blending requirement: {tokens[i:i + 4]}')
        else:
            i += 1
m = gp.Model('wine_blend')
x_vars = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
for b in brands:
    m.addConstr(y_vars[b] == gp.quicksum((x_vars[g, b] for g in grades)), name=f'ydef_{b}')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
eps = 1e-06
for ((g, b), ub) in blend_ub.items():
    m.addConstr(x_vars[g, b] <= ub * y_vars[b] - eps, name=f'blendub_{g}_{b}')
for ((g, b), lb) in blend_lb.items():
    m.addConstr(x_vars[g, b] >= lb * y_vars[b] + eps, name=f'blendlb_{g}_{b}')
if 'Red' in brands:
    m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
revenue = gp.quicksum((brand_price[b] * y_vars[b] for b in brands))
cost = gp.quicksum((grade_cost[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(revenue - cost, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')