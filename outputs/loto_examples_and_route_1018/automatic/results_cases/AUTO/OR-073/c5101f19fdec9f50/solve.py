LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A1","Product I":"5","Product II":"10","Product III":"","Available Equipment Operating Time":"6000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A2","Product I":"7","Product II":"9","Product III":"12","Available Equipment Operating Time":"10000","Equipment Cost at Full Load (yuan)":"321"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A3","Product I":"6","Product II":"11","Product III":"2","Available Equipment Operating Time":"8000","Equipment Cost at Full Load (yuan)":"203"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B1","Product I":"6","Product II":"8","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"250"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B2","Product I":"4","Product II":"","Product III":"11","Available Equipment Operating Time":"7000","Equipment Cost at Full Load (yuan)":"783"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B3","Product I":"7","Product II":"","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"200"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B4","Product I":"3","Product II":"5","Product III":"8","Available Equipment Operating Time":"5000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Raw Material Cost (yuan/unit)","Product I":"0.25","Product II":"0.35","Product III":"0.5","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Unit Price (yuan/unit)","Product I":"1.25","Product II":"2","Product III":"2.8","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
equipment_data = []
raw_material_cost = {}
unit_price = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    eq = vals['Equipment / Cost']
    if eq in ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4']:
        equipment_data.append({'equipment': eq, 'Product I': float(vals['Product I']) if vals['Product I'] != '' else None, 'Product II': float(vals['Product II']) if vals['Product II'] != '' else None, 'Product III': float(vals['Product III']) if vals['Product III'] != '' else None, 'Available Equipment Operating Time': float(vals['Available Equipment Operating Time']), 'Equipment Cost at Full Load (yuan)': float(vals['Equipment Cost at Full Load (yuan)'])})
    elif eq == 'Raw Material Cost (yuan/unit)':
        for prod in ['Product I', 'Product II', 'Product III']:
            raw_material_cost[prod] = float(vals[prod])
    elif eq == 'Unit Price (yuan/unit)':
        for prod in ['Product I', 'Product II', 'Product III']:
            unit_price[prod] = float(vals[prod])
products = ['I', 'II', 'III']
product_full = {'I': 'Product I', 'II': 'Product II', 'III': 'Product III'}
procedure_A_equip = {'I': ['A1', 'A2', 'A3'], 'II': ['A1', 'A2', 'A3'], 'III': ['A2', 'A3']}
procedure_B_equip = {'I': ['B1', 'B2', 'B3', 'B4'], 'II': ['B1', 'B4'], 'III': ['B2', 'B4']}
processing_time = {}
equipment_capacity = {}
equipment_cost = {}
for eq in equipment_data:
    eqid = eq['equipment']
    equipment_capacity[eqid] = eq['Available Equipment Operating Time']
    equipment_cost[eqid] = eq['Equipment Cost at Full Load (yuan)']
    for p in products:
        pf = product_full[p]
        if eq[pf] is not None:
            processing_time[eqid, p] = eq[pf]
m = gp.Model('factory_production')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y_A_vars = {}
for p in products:
    for eq in procedure_A_equip.get(p, []):
        y_A_vars[eq, p] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'y_{eq}_{p}_A')
y_B_vars = {}
for p in products:
    for eq in procedure_B_equip.get(p, []):
        y_B_vars[eq, p] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'y_{eq}_{p}_B')
m.update()
m.addConstr(x_vars['I'] == gp.quicksum((y_A_vars[eq, 'I'] for eq in procedure_A_equip['I'])), name='assign_A_I')
m.addConstr(x_vars['II'] == gp.quicksum((y_A_vars[eq, 'II'] for eq in procedure_A_equip['II'])), name='assign_A_II')
m.addConstr(x_vars['III'] == gp.quicksum((y_A_vars[eq, 'III'] for eq in procedure_A_equip['III'])), name='assign_A_III')
m.addConstr(x_vars['I'] == gp.quicksum((y_B_vars[eq, 'I'] for eq in procedure_B_equip['I'])), name='assign_B_I')
m.addConstr(x_vars['II'] == gp.quicksum((y_B_vars[eq, 'II'] for eq in procedure_B_equip['II'])), name='assign_B_II')
m.addConstr(x_vars['III'] == gp.quicksum((y_B_vars[eq, 'III'] for eq in procedure_B_equip['III'])), name='assign_B_III')
for eq in equipment_capacity:
    expr = gp.LinExpr()
    for p in products:
        if (eq, p) in y_A_vars and (eq, p) in processing_time:
            expr += processing_time[eq, p] * y_A_vars[eq, p]
    for p in products:
        if (eq, p) in y_B_vars and (eq, p) in processing_time:
            expr += processing_time[eq, p] * y_B_vars[eq, p]
    m.addConstr(expr <= equipment_capacity[eq], name=f'cap_{eq}')
revenue = gp.quicksum((unit_price[product_full[p]] * x_vars[p] for p in products))
raw_cost = gp.quicksum((raw_material_cost[product_full[p]] * x_vars[p] for p in products))
equip_cost_expr = gp.LinExpr()
for eq in equipment_capacity:
    eq_cost = equipment_cost[eq]
    eq_cap = equipment_capacity[eq]
    for p in products:
        if (eq, p) in y_A_vars and (eq, p) in processing_time:
            equip_cost_expr += eq_cost / eq_cap * processing_time[eq, p] * y_A_vars[eq, p]
    for p in products:
        if (eq, p) in y_B_vars and (eq, p) in processing_time:
            equip_cost_expr += eq_cost / eq_cap * processing_time[eq, p] * y_B_vars[eq, p]
m.setObjective(revenue - raw_cost - equip_cost_expr, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')