LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv", "values": {"Equipment / Cost": "A1", "Product I": "5", "Product II": "10", "Product III": "", "Available Equipment Operating Time": "6000", "Equipment Cost at Full Load (yuan)": "300"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv", "values": {"Equipment / Cost": "A2", "Product I": "7", "Product II": "9", "Product III": "12", "Available Equipment Operating Time": "10000", "Equipment Cost at Full Load (yuan)": "321"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv", "values": {"Equipment / Cost": "A3", "Product I": "6", "Product II": "11", "Product III": "2", "Available Equipment Operating Time": "8000", "Equipment Cost at Full Load (yuan)": "203"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv", "values": {"Equipment / Cost": "B1", "Product I": "6", "Product II": "8", "Product III": "", "Available Equipment Operating Time": "4000", "Equipment Cost at Full Load (yuan)": "250"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv", "values": {"Equipment / Cost": "B2", "Product I": "4", "Product II": "", "Product III": "11", "Available Equipment Operating Time": "7000", "Equipment Cost at Full Load (yuan)": "783"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv", "values": {"Equipment / Cost": "B3", "Product I": "7", "Product II": "", "Product III": "", "Available Equipment Operating Time": "4000", "Equipment Cost at Full Load (yuan)": "200"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv", "values": {"Equipment / Cost": "B4", "Product I": "3", "Product II": "5", "Product III": "8", "Available Equipment Operating Time": "5000", "Equipment Cost at Full Load (yuan)": "300"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv", "values": {"Equipment / Cost": "Raw Material Cost (yuan/unit)", "Product I": "0.25", "Product II": "0.35", "Product III": "0.5", "Available Equipment Operating Time": "", "Equipment Cost at Full Load (yuan)": ""}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv", "values": {"Equipment / Cost": "Unit Price (yuan/unit)", "Product I": "1.25", "Product II": "2", "Product III": "2.8", "Available Equipment Operating Time": "", "Equipment Cost at Full Load (yuan)": ""}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
records = [r for r in LEGACY_RECORDS if r['source'].endswith('43.csv')]
product_ids = ['Product I', 'Product II', 'Product III']
product_keys = ['I', 'II', 'III']
product_map = dict(zip(product_ids, product_keys))
equipment_A = ['A1', 'A2']
equipment_B = ['B1', 'B2', 'B3']
t_A = {}
t_B = {}
T_e = {}
C_e = {}
for rec in records:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq in equipment_A:
        for pid, pkey in zip(product_ids, product_keys):
            val = v[pid]
            if val != '':
                t_A[eq, pkey] = float(val)
        T_e[eq] = float(v['Available Equipment Operating Time'])
        C_e[eq] = float(v['Equipment Cost at Full Load (yuan)'])
    elif eq in equipment_B:
        for pid, pkey in zip(product_ids, product_keys):
            val = v[pid]
            if val != '':
                t_B[eq, pkey] = float(val)
        T_e[eq] = float(v['Available Equipment Operating Time'])
        C_e[eq] = float(v['Equipment Cost at Full Load (yuan)'])
raw_cost = {}
sell_price = {}
for rec in records:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq == 'Raw Material Cost (yuan/unit)':
        for pid, pkey in zip(product_ids, product_keys):
            raw_cost[pkey] = float(v[pid])
    if eq == 'Unit Price (yuan/unit)':
        for pid, pkey in zip(product_ids, product_keys):
            sell_price[pkey] = float(v[pid])
A_p = {'I': ['A1', 'A2'], 'II': ['A1', 'A2'], 'III': ['A2']}
B_p = {'I': ['B1', 'B2', 'B3'], 'II': ['B1'], 'III': ['B2']}
P_e_A = {e: [] for e in equipment_A}
for p in product_keys:
    for e in A_p[p]:
        P_e_A[e].append(p)
P_e_B = {e: [] for e in equipment_B}
for p in product_keys:
    for e in B_p[p]:
        P_e_B[e].append(p)
for e in equipment_A:
    for p in P_e_A[e]:
        if (e, p) not in t_A:
            raise ValueError(f'Missing t_A for ({e},{p})')
for e in equipment_B:
    for p in P_e_B[e]:
        if (e, p) not in t_B:
            raise ValueError(f'Missing t_B for ({e},{p})')
for p in product_keys:
    if p not in raw_cost or p not in sell_price:
        raise ValueError(f'Missing cost or price for product {p}')
for e in equipment_A + equipment_B:
    if e not in T_e or e not in C_e:
        raise ValueError(f'Missing T_e or C_e for equipment {e}')
m = gp.Model('Factory_Production_Optimization')
x_vars = m.addVars(product_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
xA_vars = m.addVars(((e, p) for e in equipment_A for p in P_e_A[e]), lb=0, vtype=GRB.CONTINUOUS, name='')
xB_vars = m.addVars(((e, p) for e in equipment_B for p in P_e_B[e]), lb=0, vtype=GRB.CONTINUOUS, name='')
for p in product_keys:
    m.addConstr(x_vars[p] == gp.quicksum((xA_vars[e, p] for e in A_p[p])), name=f'assignA_{p}')
for p in product_keys:
    m.addConstr(x_vars[p] == gp.quicksum((xB_vars[e, p] for e in B_p[p])), name=f'assignB_{p}')
for e in equipment_A:
    m.addConstr(gp.quicksum((t_A[e, p] * xA_vars[e, p] for p in P_e_A[e])) <= T_e[e], name=f'capA_{e}')
for e in equipment_B:
    m.addConstr(gp.quicksum((t_B[e, p] * xB_vars[e, p] for p in P_e_B[e])) <= T_e[e], name=f'capB_{e}')
profit_expr = gp.quicksum(((sell_price[p] - raw_cost[p]) * x_vars[p] for p in product_keys))
equip_cost_expr = gp.quicksum((C_e[e] / T_e[e] * gp.quicksum((t_A[e, p] * xA_vars[e, p] for p in P_e_A[e])) for e in equipment_A)) + gp.quicksum((C_e[e] / T_e[e] * gp.quicksum((t_B[e, p] * xB_vars[e, p] for p in P_e_B[e])) for e in equipment_B))
m.setObjective(profit_expr - equip_cost_expr, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')