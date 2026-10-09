LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "I",\n      "Daily Supply (kg)": "1500",\n      "Cost (CNY/kg)": "6"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "II",\n      "Daily Supply (kg)": "2000",\n      "Cost (CNY/kg)": "4.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "III",\n      "Daily Supply (kg)": "1000",\n      "Cost (CNY/kg)": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Red",\n      "Blending Requirements": "I less than 10%  II more than 50%",\n      "Selling Price (CNY/kg)": "5.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Yellow",\n      "Blending Requirements": "III less than 70%  I more than 20%",\n      "Selling Price (CNY/kg)": "5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Blue",\n      "Blending Requirements": "I less than 50%  II more than 10%",\n      "Selling Price (CNY/kg)": "4.8"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Red', 'Blending Requirements': 'I less than 10%  II more than 50%', 'Selling Price (CNY/kg)': '5.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Yellow', 'Blending Requirements': 'III less than 70%  I more than 20%', 'Selling Price (CNY/kg)': '5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Blue', 'Blending Requirements': 'I less than 50%  II more than 10%', 'Selling Price (CNY/kg)': '4.8'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Red', 'Blending Requirements': 'I less than 10%  II more than 50%', 'Selling Price (CNY/kg)': '5.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Yellow', 'Blending Requirements': 'III less than 70%  I more than 20%', 'Selling Price (CNY/kg)': '5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Blue', 'Blending Requirements': 'I less than 50%  II more than 10%', 'Selling Price (CNY/kg)': '4.8'}}]
grades = []
grade_cost = {}
grade_supply = {}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('30-1.csv'):
        g = rec['values']['Grade']
        grades.append(g)
        grade_cost[g] = float(rec['values']['Cost (CNY/kg)'])
        grade_supply[g] = float(rec['values']['Daily Supply (kg)'])
grades = list(dict.fromkeys(grades))
brands = []
brand_price = {}
brand_blend = {}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('30-2.csv'):
        b = rec['values']['Brand']
        brands.append(b)
        brand_price[b] = float(rec['values']['Selling Price (CNY/kg)'])
        brand_blend[b] = rec['values']['Blending Requirements']
brands = list(dict.fromkeys(brands))
blend_reqs = {b: {} for b in brands}
for b in brands:
    req = brand_blend[b]
    tokens = req.replace('%', '').split()
    i = 0
    while i < len(tokens):
        if tokens[i] in grades:
            g = tokens[i]
            if tokens[i + 1] == 'less':
                sense = '<'
                val = float(tokens[i + 3]) / 100 if tokens[i + 3].endswith('%') else float(tokens[i + 3]) / 100 if '.' in tokens[i + 3] else float(tokens[i + 3]) / 100
                i += 4
            elif tokens[i + 1] == 'more':
                sense = '>'
                val = float(tokens[i + 3]) / 100 if tokens[i + 3].endswith('%') else float(tokens[i + 3]) / 100 if '.' in tokens[i + 3] else float(tokens[i + 3]) / 100
                i += 4
            elif tokens[i + 1] == 'less' and tokens[i + 2] == 'than':
                sense = '<'
                val = float(tokens[i + 3]) / 100
                i += 4
            elif tokens[i + 1] == 'more' and tokens[i + 2] == 'than':
                sense = '>'
                val = float(tokens[i + 3]) / 100
                i += 4
            else:
                raise ValueError(f'Unknown blending requirement: {tokens[i:i + 4]}')
            blend_reqs[b][g] = (sense, val)
        else:
            i += 1
blend_reqs['Red'] = {'I': ('<', 0.1), 'II': ('>', 0.5)}
blend_reqs['Yellow'] = {'III': ('<', 0.7), 'I': ('>', 0.2)}
blend_reqs['Blue'] = {'I': ('<', 0.5), 'II': ('>', 0.1)}
m = gp.Model('wine_blending')
x_vars = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
revenue = gp.quicksum((brand_price[b] * y_vars[b] for b in brands))
cost = gp.quicksum((grade_cost[g] * gp.quicksum((x_vars[g, b] for b in brands)) for g in grades))
m.setObjective(revenue - cost, GRB.MAXIMIZE)
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
for b in brands:
    m.addConstr(y_vars[b] == gp.quicksum((x_vars[g, b] for g in grades)), name=f'ydef_{b}')
m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
for b in brands:
    for g in blend_reqs[b]:
        (sense, val) = blend_reqs[b][g]
        if sense == '<':
            m.addConstr(x_vars[g, b] <= val * y_vars[b] - 1e-06, name=f'blend_{b}_{g}_lt')
        elif sense == '>':
            m.addConstr(x_vars[g, b] >= val * y_vars[b] + 1e-06, name=f'blend_{b}_{g}_gt')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')