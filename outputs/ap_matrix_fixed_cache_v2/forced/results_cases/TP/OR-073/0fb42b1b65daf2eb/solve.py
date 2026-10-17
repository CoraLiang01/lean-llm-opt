LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A1","Product I":"5","Product II":"10","Product III":"","Available Equipment Operating Time":"6000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A2","Product I":"7","Product II":"9","Product III":"12","Available Equipment Operating Time":"10000","Equipment Cost at Full Load (yuan)":"321"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A3","Product I":"6","Product II":"11","Product III":"2","Available Equipment Operating Time":"8000","Equipment Cost at Full Load (yuan)":"203"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B1","Product I":"6","Product II":"8","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"250"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B2","Product I":"4","Product II":"","Product III":"11","Available Equipment Operating Time":"7000","Equipment Cost at Full Load (yuan)":"783"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B3","Product I":"7","Product II":"","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"200"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B4","Product I":"3","Product II":"5","Product III":"8","Available Equipment Operating Time":"5000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Raw Material Cost (yuan/unit)","Product I":"0.25","Product II":"0.35","Product III":"0.5","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Unit Price (yuan/unit)","Product I":"1.25","Product II":"2","Product III":"2.8","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
records = [r for r in LEGACY_RECORDS if r['source'] == '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv']
equipment_list = []
product_list = []
proc_time = {}
equip_time = {}
equip_cost = {}
for rec in records:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq not in ['Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)']:
        equipment_list.append(eq)
        for p in ['Product I', 'Product II', 'Product III']:
            if v[p] != '':
                proc_time[eq, p] = float(v[p])
        equip_time[eq] = float(v['Available Equipment Operating Time'])
        equip_cost[eq] = float(v['Equipment Cost at Full Load (yuan)'])
equipment_list = list(dict.fromkeys(equipment_list))
for p in ['Product I', 'Product II', 'Product III']:
    product_list.append(p)
raw_material_cost = {}
unit_price = {}
for rec in records:
    v = rec['values']
    if v['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)':
        for p in product_list:
            raw_material_cost[p] = float(v[p])
    if v['Equipment / Cost'] == 'Unit Price (yuan/unit)':
        for p in product_list:
            unit_price[p] = float(v[p])
allowed_pairs = list(proc_time.keys())
procedure_A = [e for e in equipment_list if e.startswith('A')]
procedure_B = [e for e in equipment_list if e.startswith('B')]
A_assign = {p: [e for e in procedure_A if (e, p) in allowed_pairs] for p in product_list}
B_assign = {p: [e for e in procedure_B if (e, p) in allowed_pairs] for p in product_list}
m = gp.Model('factory_opt')
x_vars = m.addVars(product_list, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(allowed_pairs, lb=0, vtype=GRB.CONTINUOUS, name='')
profit_expr = gp.LinExpr()
for p in product_list:
    profit_expr += (unit_price[p] - raw_material_cost[p]) * x_vars[p]
for e in equipment_list:
    numer = gp.LinExpr()
    for p in product_list:
        if (e, p) in proc_time:
            numer += proc_time[e, p] * y_vars[e, p]
    profit_expr -= equip_cost[e] * numer / equip_time[e]
m.setObjective(profit_expr, GRB.MAXIMIZE)
m.addConstr(y_vars['A1', 'Product I'] + y_vars['A2', 'Product I'] + y_vars['A3', 'Product I'] == x_vars['Product I'], name='A_PI')
m.addConstr(y_vars['A1', 'Product II'] + y_vars['A2', 'Product II'] + y_vars['A3', 'Product II'] == x_vars['Product II'], name='A_PII')
m.addConstr(y_vars['A2', 'Product III'] + y_vars['A3', 'Product III'] == x_vars['Product III'], name='A_PIII')
m.addConstr(y_vars['B1', 'Product I'] + y_vars['B2', 'Product I'] + y_vars['B3', 'Product I'] + y_vars['B4', 'Product I'] == x_vars['Product I'], name='B_PI')
m.addConstr(y_vars['B1', 'Product II'] + y_vars['B4', 'Product II'] == x_vars['Product II'], name='B_PII')
m.addConstr(y_vars['B2', 'Product III'] + y_vars['B4', 'Product III'] == x_vars['Product III'], name='B_PIII')
m.addConstr(5 * y_vars['A1', 'Product I'] + 10 * y_vars['A1', 'Product II'] <= 6000, name='cap_A1')
m.addConstr(7 * y_vars['A2', 'Product I'] + 9 * y_vars['A2', 'Product II'] + 12 * y_vars['A2', 'Product III'] <= 10000, name='cap_A2')
m.addConstr(6 * y_vars['A3', 'Product I'] + 11 * y_vars['A3', 'Product II'] + 2 * y_vars['A3', 'Product III'] <= 8000, name='cap_A3')
m.addConstr(6 * y_vars['B1', 'Product I'] + 8 * y_vars['B1', 'Product II'] <= 4000, name='cap_B1')
m.addConstr(4 * y_vars['B2', 'Product I'] + 11 * y_vars['B2', 'Product III'] <= 7000, name='cap_B2')
m.addConstr(7 * y_vars['B3', 'Product I'] <= 4000, name='cap_B3')
m.addConstr(3 * y_vars['B4', 'Product I'] + 5 * y_vars['B4', 'Product II'] + 8 * y_vars['B4', 'Product III'] <= 5000, name='cap_B4')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')