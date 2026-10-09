LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A1","Product I":"5","Product II":"10","Product III":"","Available Equipment Operating Time":"6000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A2","Product I":"7","Product II":"9","Product III":"12","Available Equipment Operating Time":"10000","Equipment Cost at Full Load (yuan)":"321"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A3","Product I":"6","Product II":"11","Product III":"2","Available Equipment Operating Time":"8000","Equipment Cost at Full Load (yuan)":"203"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B1","Product I":"6","Product II":"8","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"250"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B2","Product I":"4","Product II":"","Product III":"11","Available Equipment Operating Time":"7000","Equipment Cost at Full Load (yuan)":"783"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B3","Product I":"7","Product II":"","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"200"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B4","Product I":"3","Product II":"5","Product III":"8","Available Equipment Operating Time":"5000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Raw Material Cost (yuan/unit)","Product I":"0.25","Product II":"0.35","Product III":"0.5","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Unit Price (yuan/unit)","Product I":"1.25","Product II":"2","Product III":"2.8","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
records = [r for r in LEGACY_RECORDS if r['source'] == '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv']
equipments = []
products = []
proc_time = {}
equip_time = {}
equip_cost = {}
for rec in records:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq in ['Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)']:
        continue
    equipments.append(eq)
    for p in ['Product I', 'Product II', 'Product III']:
        if v[p] != '':
            proc_time[eq, p] = float(v[p])
            if p not in products:
                products.append(p)
    equip_time[eq] = float(v['Available Equipment Operating Time'])
    equip_cost[eq] = float(v['Equipment Cost at Full Load (yuan)'])
raw_material_cost = {}
unit_price = {}
for rec in records:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq == 'Raw Material Cost (yuan/unit)':
        for p in ['Product I', 'Product II', 'Product III']:
            raw_material_cost[p] = float(v[p])
    if eq == 'Unit Price (yuan/unit)':
        for p in ['Product I', 'Product II', 'Product III']:
            unit_price[p] = float(v[p])
procedure_A = [e for e in equipments if e.startswith('A')]
procedure_B = [e for e in equipments if e.startswith('B')]
y_pairs = []
for eq in equipments:
    for p in products:
        if (eq, p) in proc_time:
            y_pairs.append((eq, p))
m = gp.Model('factory_opt')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(y_pairs, lb=0, vtype=GRB.CONTINUOUS, name='')
m.addConstr(gp.quicksum((y_vars[e, 'Product I'] for e in procedure_A if (e, 'Product I') in y_vars)) == x_vars['Product I'], name='fb_PI_A')
m.addConstr(gp.quicksum((y_vars[e, 'Product I'] for e in procedure_B if (e, 'Product I') in y_vars)) == x_vars['Product I'], name='fb_PI_B')
m.addConstr(gp.quicksum((y_vars[e, 'Product II'] for e in procedure_A if (e, 'Product II') in y_vars)) == x_vars['Product II'], name='fb_PII_A')
m.addConstr(gp.quicksum((y_vars[e, 'Product II'] for e in procedure_B if (e, 'Product II') in y_vars)) == x_vars['Product II'], name='fb_PII_B')
m.addConstr(gp.quicksum((y_vars[e, 'Product III'] for e in procedure_A if (e, 'Product III') in y_vars)) == x_vars['Product III'], name='fb_PIII_A')
m.addConstr(gp.quicksum((y_vars[e, 'Product III'] for e in procedure_B if (e, 'Product III') in y_vars)) == x_vars['Product III'], name='fb_PIII_B')
for eq in equipments:
    m.addConstr(gp.quicksum((proc_time[eq, p] * y_vars[eq, p] for p in products if (eq, p) in y_vars)) <= equip_time[eq], name=f'cap_{eq}')
revenue = gp.quicksum((unit_price[p] * x_vars[p] for p in products))
raw_cost = gp.quicksum((raw_material_cost[p] * x_vars[p] for p in products))
equip_cost_expr = gp.quicksum((equip_cost[eq] * (gp.quicksum((proc_time[eq, p] * y_vars[eq, p] for p in products if (eq, p) in y_vars)) / equip_time[eq]) for eq in equipments))
m.setObjective(revenue - raw_cost - equip_cost_expr, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')