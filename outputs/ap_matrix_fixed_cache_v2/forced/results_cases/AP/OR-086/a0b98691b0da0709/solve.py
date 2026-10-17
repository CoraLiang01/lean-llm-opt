LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "I",\n      "Daily Supply (kg)": "1500",\n      "Cost (CNY/kg)": "6"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "II",\n      "Daily Supply (kg)": "2000",\n      "Cost (CNY/kg)": "4.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv",\n    "values": {\n      "Grade": "III",\n      "Daily Supply (kg)": "1000",\n      "Cost (CNY/kg)": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Red",\n      "Blending Requirements": "I less than 10%  II more than 50%",\n      "Selling Price (CNY/kg)": "5.5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Yellow",\n      "Blending Requirements": "III less than 70%  I more than 20%",\n      "Selling Price (CNY/kg)": "5"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv",\n    "values": {\n      "Brand": "Blue",\n      "Blending Requirements": "I less than 50%  II more than 10%",\n      "Selling Price (CNY/kg)": "4.8"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Red', 'Blending Requirements': 'I less than 10%  II more than 50%', 'Selling Price (CNY/kg)': '5.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Yellow', 'Blending Requirements': 'III less than 70%  I more than 20%', 'Selling Price (CNY/kg)': '5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Blue', 'Blending Requirements': 'I less than 50%  II more than 10%', 'Selling Price (CNY/kg)': '4.8'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'I', 'Daily Supply (kg)': '1500', 'Cost (CNY/kg)': '6'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'II', 'Daily Supply (kg)': '2000', 'Cost (CNY/kg)': '4.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv', 'values': {'Grade': 'III', 'Daily Supply (kg)': '1000', 'Cost (CNY/kg)': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Red', 'Blending Requirements': 'I less than 10%  II more than 50%', 'Selling Price (CNY/kg)': '5.5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Yellow', 'Blending Requirements': 'III less than 70%  I more than 20%', 'Selling Price (CNY/kg)': '5'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv', 'values': {'Brand': 'Blue', 'Blending Requirements': 'I less than 50%  II more than 10%', 'Selling Price (CNY/kg)': '4.8'}}]
raw_grades = []
raw_grade_supply = {}
raw_grade_cost = {}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('30-1.csv'):
        g = rec['values']['Grade']
        raw_grades.append(g)
        raw_grade_supply[g] = float(rec['values']['Daily Supply (kg)'])
        raw_grade_cost[g] = float(rec['values']['Cost (CNY/kg)'])
brands = []
brand_price = {}
blending_reqs = {}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('30-2.csv'):
        b = rec['values']['Brand']
        brands.append(b)
        brand_price[b] = float(rec['values']['Selling Price (CNY/kg)'])
        reqs = rec['values']['Blending Requirements']
        blending_reqs[b] = {}
        for part in reqs.split():
            if part in raw_grades:
                g = part
                continue
            if part == 'less':
                bound_type = 'upper'
                continue
            if part == 'more':
                bound_type = 'lower'
                continue
            if part.endswith('%'):
                val = float(part[:-1]) / 100.0
                blending_reqs[b][g] = blending_reqs[b].get(g, {})
                blending_reqs[b][g][bound_type] = val
m = gp.Model('wine_blend')
x_vars = m.addVars(raw_grades, brands, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(brands, lb=0, vtype=GRB.CONTINUOUS, name='')
flag_vars = m.addVars(brands, vtype=GRB.BINARY, name='')
revenue = gp.quicksum((brand_price[b] * y_vars[b] for b in brands))
cost = gp.quicksum((raw_grade_cost[g] * x_vars[g, b] for g in raw_grades for b in brands))
m.setObjective(revenue - cost, GRB.MAXIMIZE)
for b in brands:
    m.addConstr(y_vars[b] == gp.quicksum((x_vars[g, b] for g in raw_grades)), name=f'ydef_{b}')
for g in raw_grades:
    m.addConstr(gp.quicksum((x_vars[g, b] for b in brands)) <= raw_grade_supply[g], name=f'supply_{g}')
m.addConstr(y_vars['Red'] >= 2000, name='minprod_Red')
bigM = sum(raw_grade_supply.values())
eps = 0.001
for b in brands:
    m.addConstr(y_vars[b] <= bigM * flag_vars[b], name=f'flagub_{b}')
    m.addConstr(y_vars[b] >= eps * flag_vars[b], name=f'flaglb_{b}')
    for g in raw_grades:
        m.addConstr(x_vars[g, b] <= bigM * flag_vars[b], name=f'xub_{g}_{b}')
        m.addConstr(x_vars[g, b] >= 0, name=f'xlb_{g}_{b}')
for b in brands:
    if b in blending_reqs:
        for g in blending_reqs[b]:
            req = blending_reqs[b][g]
            if 'upper' in req:
                m.addConstr(x_vars[g, b] <= (req['upper'] - 1e-06) * y_vars[b] + bigM * (1 - flag_vars[b]), name=f'blend_ub_{g}_{b}')
            if 'lower' in req:
                m.addConstr(x_vars[g, b] >= (req['lower'] + 1e-06) * y_vars[b] - bigM * (1 - flag_vars[b]), name=f'blend_lb_{g}_{b}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')