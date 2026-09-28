LEGACY_OBSERVATION = '{"values": {"Equipment / Cost": "A1", "Product I": "5", "Product II": "10", "Product III": "", "Available Equipment Operating Time": "6000", "Equipment Cost at Full Load (yuan)": "300"}}\n{"values": {"Equipment / Cost": "A2", "Product I": "7", "Product II": "9", "Product III": "12", "Available Equipment Operating Time": "10000", "Equipment Cost at Full Load (yuan)": "321"}}\n{"values": {"Equipment / Cost": "A3", "Product I": "6", "Product II": "11", "Product III": "2", "Available Equipment Operating Time": "8000", "Equipment Cost at Full Load (yuan)": "203"}}\n{"values": {"Equipment / Cost": "B1", "Product I": "6", "Product II": "8", "Product III": "", "Available Equipment Operating Time": "4000", "Equipment Cost at Full Load (yuan)": "250"}}\n{"values": {"Equipment / Cost": "B2", "Product I": "4", "Product II": "", "Product III": "11", "Available Equipment Operating Time": "7000", "Equipment Cost at Full Load (yuan)": "783"}}\n{"values": {"Equipment / Cost": "B3", "Product I": "7", "Product II": "", "Product III": "", "Available Equipment Operating Time": "4000", "Equipment Cost at Full Load (yuan)": "200"}}\n{"values": {"Equipment / Cost": "B4", "Product I": "3", "Product II": "5", "Product III": "8", "Available Equipment Operating Time": "5000", "Equipment Cost at Full Load (yuan)": "300"}}\n{"values": {"Equipment / Cost": "Raw Material Cost (yuan/unit)", "Product I": "0.25", "Product II": "0.35", "Product III": "0.5", "Available Equipment Operating Time": "", "Equipment Cost at Full Load (yuan)": ""}}\n{"values": {"Equipment / Cost": "Unit Price (yuan/unit)", "Product I": "1.25", "Product II": "2", "Product III": "2.8", "Available Equipment Operating Time": "", "Equipment Cost at Full Load (yuan)": ""}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
equipments = []
products = []
proc_time = {}
avail_time = {}
equip_cost = {}
for rec in records:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq not in ['Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)']:
        equipments.append(eq)
        for p in ['Product I', 'Product II', 'Product III']:
            if v[p] != '':
                proc_time[eq, p] = float(v[p])
        avail_time[eq] = float(v['Available Equipment Operating Time'])
        equip_cost[eq] = float(v['Equipment Cost at Full Load (yuan)'])
    else:
        for p in ['Product I', 'Product II', 'Product III']:
            if p not in products:
                products.append(p)
equipments = [e for e in equipments if e not in ['Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)']]
raw_cost = {}
sell_price = {}
for rec in records:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq == 'Raw Material Cost (yuan/unit)':
        for p in ['Product I', 'Product II', 'Product III']:
            raw_cost[p] = float(v[p])
    if eq == 'Unit Price (yuan/unit)':
        for p in ['Product I', 'Product II', 'Product III']:
            sell_price[p] = float(v[p])
allowed = []
for (eq, p), t in proc_time.items():
    allowed.append((eq, p))
A_equips = [e for e in equipments if e.startswith('A')]
B_equips = [e for e in equipments if e.startswith('B')]
A_allowed = {}
B_allowed = {}
for p in products:
    A_allowed[p] = [e for e in A_equips if (e, p) in allowed]
    B_allowed[p] = [e for e in B_equips if (e, p) in allowed]
m = gp.Model('factory_opt')
x = m.addVars(allowed, lb=0, vtype=GRB.CONTINUOUS, name='')
Q = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
for p in products:
    m.addConstr(Q[p] == gp.quicksum((x[e, p] for e in A_allowed[p])), name=f'flowA_{p}')
    m.addConstr(Q[p] == gp.quicksum((x[e, p] for e in B_allowed[p])), name=f'flowB_{p}')
for e in equipments:
    expr = gp.LinExpr()
    for p in products:
        if (e, p) in proc_time:
            expr += proc_time[e, p] * x[e, p]
    m.addConstr(expr <= avail_time[e], name=f'cap_{e}')
for e in equipments:
    for p in products:
        if (e, p) not in allowed:
            pass
profit = gp.quicksum((sell_price[p] * Q[p] for p in products))
rawmat = gp.quicksum((raw_cost[p] * Q[p] for p in products))
equipc = gp.LinExpr()
for e in equipments:
    numer = gp.LinExpr()
    for p in products:
        if (e, p) in proc_time:
            numer += proc_time[e, p] * x[e, p]
    equipc += equip_cost[e] * numer / avail_time[e]
m.setObjective(profit - rawmat - equipc, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')