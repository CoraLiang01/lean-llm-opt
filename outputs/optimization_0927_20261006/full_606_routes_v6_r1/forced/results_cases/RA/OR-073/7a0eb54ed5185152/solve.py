LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A1","Product I":"5","Product II":"10","Product III":"","Available Equipment Operating Time":"6000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A2","Product I":"7","Product II":"9","Product III":"12","Available Equipment Operating Time":"10000","Equipment Cost at Full Load (yuan)":"321"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"A3","Product I":"6","Product II":"11","Product III":"2","Available Equipment Operating Time":"8000","Equipment Cost at Full Load (yuan)":"203"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B1","Product I":"6","Product II":"8","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"250"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B2","Product I":"4","Product II":"","Product III":"11","Available Equipment Operating Time":"7000","Equipment Cost at Full Load (yuan)":"783"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B3","Product I":"7","Product II":"","Product III":"","Available Equipment Operating Time":"4000","Equipment Cost at Full Load (yuan)":"200"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"B4","Product I":"3","Product II":"5","Product III":"8","Available Equipment Operating Time":"5000","Equipment Cost at Full Load (yuan)":"300"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Raw Material Cost (yuan/unit)","Product I":"0.25","Product II":"0.35","Product III":"0.5","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv","values":{"Equipment / Cost":"Unit Price (yuan/unit)","Product I":"1.25","Product II":"2","Product III":"2.8","Available Equipment Operating Time":"","Equipment Cost at Full Load (yuan)":""}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
records = [rec for rec in LEGACY_RECORDS if rec['source'] and rec['values']]
equipment_list = []
product_list = []
t_ei = {}
T_e = {}
C_e = {}
for rec in records:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq not in ['Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)']:
        equipment_list.append(eq)
        for prod in ['Product I', 'Product II', 'Product III']:
            if v[prod] != '':
                t_ei[eq, prod] = float(v[prod])
        T_e[eq] = float(v['Available Equipment Operating Time'])
        C_e[eq] = float(v['Equipment Cost at Full Load (yuan)'])
equipment_list = [e for e in equipment_list if e not in ['Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)']]
equipment_list = list(dict.fromkeys(equipment_list))
for rec in records:
    v = rec['values']
    if v['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)':
        product_list = [k for k in ['Product I', 'Product II', 'Product III'] if v[k] != '']
        r_i = {k: float(v[k]) for k in product_list}
    if v['Equipment / Cost'] == 'Unit Price (yuan/unit)':
        p_i = {k: float(v[k]) for k in product_list}
eligible_pairs = list(t_ei.keys())
E_i = {prod: [e for (e, p) in eligible_pairs if p == prod] for prod in product_list}
I_e = {e: [p for (eq, p) in eligible_pairs if eq == e] for e in equipment_list}
m = gp.Model('factory_production')
x_vars = m.addVars(product_list, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(eligible_pairs, lb=0, vtype=GRB.CONTINUOUS, name='')
profit_expr = gp.quicksum(((p_i[prod] - r_i[prod]) * x_vars[prod] for prod in product_list))
equipment_cost_expr = gp.quicksum((C_e[e] * gp.quicksum((t_ei[e, p] * y_vars[e, p] for p in I_e[e])) / T_e[e] for e in equipment_list))
m.setObjective(profit_expr - equipment_cost_expr, GRB.MAXIMIZE)
for prod in product_list:
    m.addConstr(x_vars[prod] == gp.quicksum((y_vars[e, prod] for e in E_i[prod])), name=f'prod_balance_{prod}')
for e in equipment_list:
    m.addConstr(gp.quicksum((t_ei[e, p] * y_vars[e, p] for p in I_e[e])) <= T_e[e], name=f'equip_time_{e}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')