LEGACY_OBSERVATION = '{"values": {"Equipment / Cost": "A1", "Product I": "5", "Product II": "10", "Product III": "", "Available Equipment Operating Time": "6000", "Equipment Cost at Full Load (yuan)": "300"}}\n{"values": {"Equipment / Cost": "A2", "Product I": "7", "Product II": "9", "Product III": "12", "Available Equipment Operating Time": "10000", "Equipment Cost at Full Load (yuan)": "321"}}\n{"values": {"Equipment / Cost": "A3", "Product I": "6", "Product II": "11", "Product III": "2", "Available Equipment Operating Time": "8000", "Equipment Cost at Full Load (yuan)": "203"}}\n{"values": {"Equipment / Cost": "B1", "Product I": "6", "Product II": "8", "Product III": "", "Available Equipment Operating Time": "4000", "Equipment Cost at Full Load (yuan)": "250"}}\n{"values": {"Equipment / Cost": "B2", "Product I": "4", "Product II": "", "Product III": "11", "Available Equipment Operating Time": "7000", "Equipment Cost at Full Load (yuan)": "783"}}\n{"values": {"Equipment / Cost": "B3", "Product I": "7", "Product II": "", "Product III": "", "Available Equipment Operating Time": "4000", "Equipment Cost at Full Load (yuan)": "200"}}\n{"values": {"Equipment / Cost": "B4", "Product I": "3", "Product II": "5", "Product III": "8", "Available Equipment Operating Time": "5000", "Equipment Cost at Full Load (yuan)": "300"}}\n{"values": {"Equipment / Cost": "Raw Material Cost (yuan/unit)", "Product I": "0.25", "Product II": "0.35", "Product III": "0.5", "Available Equipment Operating Time": "", "Equipment Cost at Full Load (yuan)": ""}}\n{"values": {"Equipment / Cost": "Unit Price (yuan/unit)", "Product I": "1.25", "Product II": "2", "Product III": "2.8", "Available Equipment Operating Time": "", "Equipment Cost at Full Load (yuan)": ""}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
equipment = []
products = []
t = {}
T = {}
C = {}
c = {}
s = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq in ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4']:
        equipment.append(eq)
        for prod in ['Product I', 'Product II', 'Product III']:
            if prod not in products and v[prod] not in [None, '']:
                products.append(prod)
            if v[prod] not in [None, '']:
                t.setdefault(eq, {})[prod] = float(v[prod])
        T[eq] = float(v['Available Equipment Operating Time'])
        C[eq] = float(v['Equipment Cost at Full Load (yuan)'])
    elif eq == 'Raw Material Cost (yuan/unit)':
        for prod in ['Product I', 'Product II', 'Product III']:
            c[prod] = float(v[prod])
    elif eq == 'Unit Price (yuan/unit)':
        for prod in ['Product I', 'Product II', 'Product III']:
            s[prod] = float(v[prod])
prod_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
products = ['Product I', 'Product II', 'Product III']
allowed = []
for e in equipment:
    for i in products:
        if e in t and i in t[e]:
            allowed.append((e, i))
A = [e for e in equipment if e.startswith('A')]
B = [e for e in equipment if e.startswith('B')]
A_proc = {'Product I': [e for e in A if (e, 'Product I') in allowed], 'Product II': [e for e in A if (e, 'Product II') in allowed], 'Product III': [e for e in A if (e, 'Product III') in allowed]}
B_proc = {'Product I': [e for e in B if (e, 'Product I') in allowed], 'Product II': [e for e in B if (e, 'Product II') in allowed], 'Product III': [e for e in B if (e, 'Product III') in allowed]}
m = gp.Model('factory_opt')
x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(allowed, lb=0, vtype=GRB.CONTINUOUS, name='')
m.addConstr(x['Product I'] == gp.quicksum((y[e, 'Product I'] for e in A_proc['Product I'])), name='A_I')
m.addConstr(x['Product II'] == gp.quicksum((y[e, 'Product II'] for e in A_proc['Product II'])), name='A_II')
m.addConstr(x['Product III'] == gp.quicksum((y[e, 'Product III'] for e in A_proc['Product III'])), name='A_III')
m.addConstr(x['Product I'] == gp.quicksum((y[e, 'Product I'] for e in B_proc['Product I'])), name='B_I')
m.addConstr(x['Product II'] == gp.quicksum((y[e, 'Product II'] for e in B_proc['Product II'])), name='B_II')
m.addConstr(x['Product III'] == gp.quicksum((y[e, 'Product III'] for e in B_proc['Product III'])), name='B_III')
for e in equipment:
    expr = gp.LinExpr()
    for i in products:
        if (e, i) in allowed:
            expr += t[e][i] * y[e, i]
    m.addConstr(expr <= T[e], name=f'cap_{e}')
profit = gp.LinExpr()
for i in products:
    profit += (s[i] - c[i]) * x[i]
equip_cost = gp.LinExpr()
for e in equipment:
    denom = T[e]
    numer = C[e]
    for i in products:
        if (e, i) in allowed:
            equip_cost += numer / denom * t[e][i] * y[e, i]
m.setObjective(profit - equip_cost, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')