LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "I",\n      "Daily Supply (kg)": "1500",\n      "Cost (CNY/kg)": "6"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "II",\n      "Daily Supply (kg)": "2000",\n      "Cost (CNY/kg)": "4.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "III",\n      "Daily Supply (kg)": "1000",\n      "Cost (CNY/kg)": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Red",\n      "Blending Requirements": "I less than 10%  II more than 50%",\n      "Selling Price (CNY/kg)": "5.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Yellow",\n      "Blending Requirements": "III less than 70%  I more than 20%",\n      "Selling Price (CNY/kg)": "5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Blue",\n      "Blending Requirements": "I less than 50%  II more than 10%",\n      "Selling Price (CNY/kg)": "4.8"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Red', 'Blending Requirements': 'I less than 10%  II more than 50%', 'Selling Price (CNY/kg)': '5.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Yellow', 'Blending Requirements': 'III less than 70%  I more than 20%', 'Selling Price (CNY/kg)': '5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Blue', 'Blending Requirements': 'I less than 50%  II more than 10%', 'Selling Price (CNY/kg)': '4.8'}}]
import gurobipy as gp
from gurobipy import GRB
grades = []
grade_supply = {}
grade_cost = {}
brands = []
brand_price = {}
blending_reqs = {}
for rec in LEGACY_RECORDS:
    src = rec['source']
    vals = rec['values']
    if src.endswith('30-1.csv'):
        g = vals['Grade']
        grades.append(g)
        grade_supply[g] = float(vals['Daily Supply (kg)'])
        grade_cost[g] = float(vals['Cost (CNY/kg)'])
    elif src.endswith('30-2.csv'):
        b = vals['Brand']
        brands.append(b)
        brand_price[b] = float(vals['Selling Price (CNY/kg)'])
        reqs = vals['Blending Requirements'].split()
        blend = []
        i = 0
        while i < len(reqs):
            if reqs[i] in grades:
                grade = reqs[i]
                if reqs[i + 1] == 'less':
                    op = '<'
                    val = float(reqs[i + 3].replace('%', '')) / 100
                    blend.append((grade, op, val))
                    i += 4
                elif reqs[i + 1] == 'more':
                    op = '>'
                    val = float(reqs[i + 3].replace('%', '')) / 100
                    blend.append((grade, op, val))
                    i += 4
                else:
                    raise ValueError(f'Unknown blending requirement: {reqs[i:i + 4]}')
            else:
                i += 1
        blending_reqs[b] = blend
grades = list(dict.fromkeys(grades))
brands = list(dict.fromkeys(brands))
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
eps = 1e-05
for b in brands:
    for g, op, val in blending_reqs[b]:
        if op == '<':
            m.addConstr(x_vars[g, b] - val * y_vars[b] <= -eps, name=f'blend_{b}_{g}_lt')
        elif op == '>':
            m.addConstr(x_vars[g, b] - val * y_vars[b] >= eps, name=f'blend_{b}_{g}_gt')
        else:
            raise ValueError(f'Unknown operator {op} in blending requirements')
if 'Red' in brands:
    m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')