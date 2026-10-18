LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "I",\n      "Daily Supply (kg)": "1500",\n      "Cost (CNY/kg)": "6"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "II",\n      "Daily Supply (kg)": "2000",\n      "Cost (CNY/kg)": "4.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "III",\n      "Daily Supply (kg)": "1000",\n      "Cost (CNY/kg)": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Red",\n      "Blending Requirements": "I less than 10%  II more than 50%",\n      "Selling Price (CNY/kg)": "5.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Yellow",\n      "Blending Requirements": "III less than 70%  I more than 20%",\n      "Selling Price (CNY/kg)": "5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Blue",\n      "Blending Requirements": "I less than 50%  II more than 10%",\n      "Selling Price (CNY/kg)": "4.8"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Red', 'Blending Requirements': 'I less than 10%  II more than 50%', 'Selling Price (CNY/kg)': '5.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Yellow', 'Blending Requirements': 'III less than 70%  I more than 20%', 'Selling Price (CNY/kg)': '5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Blue', 'Blending Requirements': 'I less than 50%  II more than 10%', 'Selling Price (CNY/kg)': '4.8'}}]
import gurobipy as gp
from gurobipy import GRB
grade_records = [r for r in LEGACY_RECORDS if r['source'].endswith('30-1.csv')]
grades = []
grade_supply = {}
grade_cost = {}
for rec in grade_records:
    g = rec['values']['Grade']
    grades.append(g)
    grade_supply[g] = float(rec['values']['Daily Supply (kg)'])
    grade_cost[g] = float(rec['values']['Cost (CNY/kg)'])
brand_records = [r for r in LEGACY_RECORDS if r['source'].endswith('30-2.csv')]
brands = []
brand_price = {}
blend_reqs = {}
for rec in brand_records:
    b = rec['values']['Brand']
    brands.append(b)
    brand_price[b] = float(rec['values']['Selling Price (CNY/kg)'])
    blend_reqs[b] = rec['values']['Blending Requirements']
blend_constraints = {'Red': [('I', '<', 0.1), ('II', '>', 0.5)], 'Yellow': [('III', '<', 0.7), ('I', '>', 0.2)], 'Blue': [('I', '<', 0.5), ('II', '>', 0.1)]}
m = gp.Model('WineBlending')
x_vars = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
for b in brands:
    m.addConstr(y_vars[b] == gp.quicksum((x_vars[g, b] for g in grades)), name=f'ydef_{b}')
epsilon = 1e-06
for b in brands:
    for g, sense, bound in blend_constraints[b]:
        if sense == '<':
            m.addConstr(x_vars[g, b] <= bound * y_vars[b] - epsilon, name=f'blend_{g}_{b}_lt')
        elif sense == '>':
            m.addConstr(x_vars[g, b] >= bound * y_vars[b] + epsilon, name=f'blend_{g}_{b}_gt')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
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