LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A1","Product I":"5","Product II":"10","Product III":"","Available Equipment Operating Time":"6000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A2","Product I":"7","Product II":"9","Product III":"12","Available Equipment Operating Time":"10000","Equipment Cost at Full Load (yuan)":"321"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A3","Product I":"6","Product II":"11","Product III":"2","Available Equipment Operating Time":"8000","Equipment Cost at Full Load (yuan)":"203"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B1","Product I":"6","Product II":"8","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"250"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B2","Product I":"4","Product II":"","Product III":"11","Available Equipment Operating Time":"7000","Equipment Cost at Full Load (yuan)":"783"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B3","Product I":"7","Product II":"","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"200"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B4","Product I":"3","Product II":"5","Product III":"8","Available Equipment Operating Time":"5000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Raw Material Cost (yuan/unit)","Product I":"0.25","Product II":"0.35","Product III":"0.5","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Unit Price (yuan/unit)","Product I":"1.25","Product II":"2","Product III":"2.8","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
equipment_list = []
product_list = []
equipment_data = {}
raw_material_cost = {}
unit_price = {}
for rec in records:
    vals = rec['values']
    eq = vals['Equipment / Cost']
    if eq not in ['Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)']:
        equipment_list.append(eq)
        equipment_data[eq] = {'Product I': vals['Product I'], 'Product II': vals['Product II'], 'Product III': vals['Product III'], 'Available Equipment Operating Time': vals['Available Equipment Operating Time'], 'Equipment Cost at Full Load (yuan)': vals['Equipment Cost at Full Load (yuan)']}
    elif eq == 'Raw Material Cost (yuan/unit)':
        for k in ['Product I', 'Product II', 'Product III']:
            raw_material_cost[k] = float(vals[k])
    elif eq == 'Unit Price (yuan/unit)':
        for k in ['Product I', 'Product II', 'Product III']:
            unit_price[k] = float(vals[k])
product_list = ['Product I', 'Product II', 'Product III']
product_idx = {'Product I': 1, 'Product II': 2, 'Product III': 3}
feasible_y = []
for eq in equipment_list:
    for k in product_list:
        t = equipment_data[eq][k]
        if t != '' and t is not None:
            feasible_y.append((eq, k))
equipment_time = {}
equipment_cost = {}
for eq in equipment_list:
    T = equipment_data[eq]['Available Equipment Operating Time']
    C = equipment_data[eq]['Equipment Cost at Full Load (yuan)']
    if T != '' and C != '':
        equipment_time[eq] = float(T)
        equipment_cost[eq] = float(C) / float(T)
    else:
        equipment_time[eq] = 0.0
        equipment_cost[eq] = 0.0
processing_time = {}
for eq in equipment_list:
    for k in product_list:
        t = equipment_data[eq][k]
        if t != '' and t is not None:
            processing_time[eq, k] = float(t)
m = gp.Model('factory_production')
x_vars = m.addVars(product_list, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(feasible_y, lb=0, vtype=GRB.CONTINUOUS, name='')
profit_expr = gp.LinExpr()
for k in product_list:
    profit_expr += (unit_price[k] - raw_material_cost[k]) * x_vars[k]
for eq in equipment_list:
    cost_sum = gp.quicksum((y_vars[eq, k] for k in product_list if (eq, k) in y_vars))
    profit_expr -= equipment_cost[eq] * cost_sum
m.setObjective(profit_expr, GRB.MAXIMIZE)
m.addConstr(x_vars['Product I'] == (y_vars['A1', 'Product I'] / processing_time['A1', 'Product I'] if ('A1', 'Product I') in y_vars else 0) + (y_vars['A2', 'Product I'] / processing_time['A2', 'Product I'] if ('A2', 'Product I') in y_vars else 0) + (y_vars['A3', 'Product I'] / processing_time['A3', 'Product I'] if ('A3', 'Product I') in y_vars else 0), name='procA_PI')
m.addConstr(x_vars['Product II'] == (y_vars['A1', 'Product II'] / processing_time['A1', 'Product II'] if ('A1', 'Product II') in y_vars else 0) + (y_vars['A2', 'Product II'] / processing_time['A2', 'Product II'] if ('A2', 'Product II') in y_vars else 0) + (y_vars['A3', 'Product II'] / processing_time['A3', 'Product II'] if ('A3', 'Product II') in y_vars else 0), name='procA_PII')
m.addConstr(x_vars['Product III'] == (y_vars['A2', 'Product III'] / processing_time['A2', 'Product III'] if ('A2', 'Product III') in y_vars else 0) + (y_vars['A3', 'Product III'] / processing_time['A3', 'Product III'] if ('A3', 'Product III') in y_vars else 0), name='procA_PIII')
m.addConstr(x_vars['Product I'] == (y_vars['B1', 'Product I'] / processing_time['B1', 'Product I'] if ('B1', 'Product I') in y_vars else 0) + (y_vars['B2', 'Product I'] / processing_time['B2', 'Product I'] if ('B2', 'Product I') in y_vars else 0) + (y_vars['B3', 'Product I'] / processing_time['B3', 'Product I'] if ('B3', 'Product I') in y_vars else 0) + (y_vars['B4', 'Product I'] / processing_time['B4', 'Product I'] if ('B4', 'Product I') in y_vars else 0), name='procB_PI')
m.addConstr(x_vars['Product II'] == (y_vars['B1', 'Product II'] / processing_time['B1', 'Product II'] if ('B1', 'Product II') in y_vars else 0), name='procB_PII')
m.addConstr(x_vars['Product III'] == (y_vars['B2', 'Product III'] / processing_time['B2', 'Product III'] if ('B2', 'Product III') in y_vars else 0) + (y_vars['B4', 'Product III'] / processing_time['B4', 'Product III'] if ('B4', 'Product III') in y_vars else 0), name='procB_PIII')
for eq in equipment_list:
    m.addConstr(gp.quicksum((y_vars[eq, k] for k in product_list if (eq, k) in y_vars)) <= equipment_time[eq], name=f'time_{eq}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')