LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A1","Product I":"5","Product II":"10","Product III":"","Available Equipment Operating Time":"6000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A2","Product I":"7","Product II":"9","Product III":"12","Available Equipment Operating Time":"10000","Equipment Cost at Full Load (yuan)":"321"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A3","Product I":"6","Product II":"11","Product III":"2","Available Equipment Operating Time":"8000","Equipment Cost at Full Load (yuan)":"203"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B1","Product I":"6","Product II":"8","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"250"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B2","Product I":"4","Product II":"","Product III":"11","Available Equipment Operating Time":"7000","Equipment Cost at Full Load (yuan)":"783"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B3","Product I":"7","Product II":"","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"200"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B4","Product I":"3","Product II":"5","Product III":"8","Available Equipment Operating Time":"5000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Raw Material Cost (yuan/unit)","Product I":"0.25","Product II":"0.35","Product III":"0.5","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Unit Price (yuan/unit)","Product I":"1.25","Product II":"2","Product III":"2.8","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
records = [r for r in LEGACY_RECORDS if r['source'] == '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv']
equipment_list = []
equipment_type = {}
equipment_proc = {}
equipment_time = {}
equipment_cost = {}
proc_map = {'A1': ('A', 1), 'A2': ('A', 2), 'B1': ('B', 1), 'B2': ('B', 2), 'B3': ('B', 3)}
for r in records:
    eq = r['values']['Equipment / Cost']
    if eq in ['A1', 'A2', 'B1', 'B2', 'B3']:
        equipment_list.append(eq)
        equipment_type[eq] = eq[0]
        equipment_proc[eq] = eq
        equipment_time[eq] = float(r['values']['Available Equipment Operating Time'])
        equipment_cost[eq] = float(r['values']['Equipment Cost at Full Load (yuan)'])
products = ['Product I', 'Product II', 'Product III']
product_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
proc_time = {}
for r in records:
    eq = r['values']['Equipment / Cost']
    if eq in ['A1', 'A2', 'B1', 'B2', 'B3']:
        for p in products:
            val = r['values'][p]
            if val != '':
                proc_time[eq, p] = float(val)
raw_cost = {}
unit_price = {}
for r in records:
    eq = r['values']['Equipment / Cost']
    if eq == 'Raw Material Cost (yuan/unit)':
        for p in products:
            raw_cost[p] = float(r['values'][p])
    if eq == 'Unit Price (yuan/unit)':
        for p in products:
            unit_price[p] = float(r['values'][p])
eligible_pairs = []
for eq in ['A1', 'A2', 'B1', 'B2', 'B3']:
    if (eq, 'Product I') in proc_time:
        eligible_pairs.append((eq, 'Product I'))
for eq in ['A1', 'A2', 'B1']:
    if (eq, 'Product II') in proc_time:
        eligible_pairs.append((eq, 'Product II'))
for eq in ['A2', 'B2']:
    if (eq, 'Product III') in proc_time:
        eligible_pairs.append((eq, 'Product III'))
m = gp.Model('factory_opt')
x_vars = m.addVars(eligible_pairs, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(['I', 'II', 'III'], lb=0, vtype=GRB.CONTINUOUS, name='')
m.addConstr(y_vars['I'] == x_vars.get(('A1', 'Product I'), 0) + x_vars.get(('A2', 'Product I'), 0), name='prodI_A')
m.addConstr(y_vars['I'] == x_vars.get(('B1', 'Product I'), 0) + x_vars.get(('B2', 'Product I'), 0) + x_vars.get(('B3', 'Product I'), 0), name='prodI_B')
m.addConstr(y_vars['II'] == x_vars.get(('A1', 'Product II'), 0) + x_vars.get(('A2', 'Product II'), 0), name='prodII_A')
m.addConstr(y_vars['II'] == x_vars.get(('B1', 'Product II'), 0), name='prodII_B')
m.addConstr(y_vars['III'] == x_vars.get(('A2', 'Product III'), 0), name='prodIII_A')
m.addConstr(y_vars['III'] == x_vars.get(('B2', 'Product III'), 0), name='prodIII_B')
for eq in ['A1', 'A2', 'B1', 'B2', 'B3']:
    m.addConstr(gp.quicksum((proc_time[eq, p] * x_vars[eq, p] for e, p in eligible_pairs if e == eq)) <= equipment_time[eq], name=f'time_{eq}')
profit_expr = 0
for p in products:
    profit_expr += (unit_price[p] - raw_cost[p]) * y_vars[product_short[p]]
for eq in ['A1', 'A2', 'B1', 'B2', 'B3']:
    eq_cost = equipment_cost[eq]
    eq_time = equipment_time[eq]
    cost_per_unit_time = eq_cost / eq_time
    for e, p in eligible_pairs:
        if e == eq:
            profit_expr -= cost_per_unit_time * proc_time[eq, p] * x_vars[eq, p]
m.setObjective(profit_expr, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')