LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "I",\n      "Daily Supply (kg)": "1500",\n      "Cost (CNY/kg)": "6"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "II",\n      "Daily Supply (kg)": "2000",\n      "Cost (CNY/kg)": "4.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "III",\n      "Daily Supply (kg)": "1000",\n      "Cost (CNY/kg)": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Red",\n      "Blending Requirements": "I less than 10%  II more than 50%",\n      "Selling Price (CNY/kg)": "5.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Yellow",\n      "Blending Requirements": "III less than 70%  I more than 20%",\n      "Selling Price (CNY/kg)": "5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Blue",\n      "Blending Requirements": "I less than 50%  II more than 10%",\n      "Selling Price (CNY/kg)": "4.8"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Red', 'Blending Requirements': 'I less than 10%  II more than 50%', 'Selling Price (CNY/kg)': '5.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Yellow', 'Blending Requirements': 'III less than 70%  I more than 20%', 'Selling Price (CNY/kg)': '5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Blue', 'Blending Requirements': 'I less than 50%  II more than 10%', 'Selling Price (CNY/kg)': '4.8'}}]
import gurobipy as gp
from gurobipy import GRB
legacy = LEGACY_RECORDS
grades = []
grade_cost = {}
grade_supply = {}
for rec in legacy:
    if rec['source'].endswith('30-1.csv'):
        g = rec['values']['Grade']
        grades.append(g)
        grade_cost[g] = float(rec['values']['Cost (CNY/kg)'])
        grade_supply[g] = float(rec['values']['Daily Supply (kg)'])
brands = []
brand_price = {}
blend_reqs = {}
for rec in legacy:
    if rec['source'].endswith('30-2.csv'):
        b = rec['values']['Brand']
        brands.append(b)
        brand_price[b] = float(rec['values']['Selling Price (CNY/kg)'])
        blend_reqs[b] = rec['values']['Blending Requirements']
if set(grades) != {'I', 'II', 'III'} or set(brands) != {'Red', 'Yellow', 'Blue'}:
    raise ValueError('Missing or extra grades/brands in LEGACY_RECORDS')
m = gp.Model('wine_blend')
x_vars = m.addVars(grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
flag_vars = m.addVars(brands, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((brand_price[b] * y_vars[b] for b in brands)) - gp.quicksum((grade_cost[g] * x_vars[g, b] for g in grades for b in brands)), GRB.MAXIMIZE)
for b in brands:
    m.addConstr(y_vars[b] == gp.quicksum((x_vars[g, b] for g in grades)), name=f'ydef_{b}')
epsilon = 0.0001
bigM = sum(grade_supply.values())
for b in brands:
    m.addConstr(y_vars[b] <= bigM * flag_vars[b], name=f'flag_ub_{b}')
    m.addConstr(y_vars[b] >= epsilon * flag_vars[b], name=f'flag_lb_{b}')
m.addConstr(x_vars['I', 'Red'] <= 0.1 * y_vars['Red'] - epsilon * flag_vars['Red'], name='blend_Red_I_ub')
m.addConstr(x_vars['II', 'Red'] >= 0.5 * y_vars['Red'] + epsilon * flag_vars['Red'], name='blend_Red_II_lb')
m.addConstr(x_vars['III', 'Yellow'] <= 0.7 * y_vars['Yellow'] - epsilon * flag_vars['Yellow'], name='blend_Yellow_III_ub')
m.addConstr(x_vars['I', 'Yellow'] >= 0.2 * y_vars['Yellow'] + epsilon * flag_vars['Yellow'], name='blend_Yellow_I_lb')
m.addConstr(x_vars['I', 'Blue'] <= 0.5 * y_vars['Blue'] - epsilon * flag_vars['Blue'], name='blend_Blue_I_ub')
m.addConstr(x_vars['II', 'Blue'] >= 0.1 * y_vars['Blue'] + epsilon * flag_vars['Blue'], name='blend_Blue_II_lb')
for g in grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= grade_supply[g], name=f'supply_{g}')
m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')