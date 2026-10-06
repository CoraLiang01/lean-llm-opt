LEGACY_OBSERVATION = 'Equipment / Cost,Product I,Product II,Product III,Available Equipment Operating Time,Equipment Cost at Full Load (yuan)\nA1,5,10,,6000,300\nA2,7,9,12,10000,321\nA3,6,11,2,8000,203\nB1,6,8,,4000,250\nB2,4,,11,7000,783\nB3,7,,,4000,200\nB4,3,5,8,5000,300\nRaw Material Cost (yuan/unit),0.25,0.35,0.5,,\nUnit Price (yuan/unit),1.25,2,2.8,,'
LEGACY_RECORDS = [{'source': '', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
equipments = []
proc_time = {}
avail_time = {}
equip_cost = {}
for rec in records:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq.startswith('A') or eq.startswith('B'):
        equipments.append(eq)
        for p in ['Product I', 'Product II', 'Product III']:
            if v[p] != '':
                if p not in products:
                    products.append(p)
                proc_time[eq, p] = float(v[p])
        avail_time[eq] = float(v['Available Equipment Operating Time'])
        equip_cost[eq] = float(v['Equipment Cost at Full Load (yuan)'])
products = sorted(products, key=lambda x: ['Product I', 'Product II', 'Product III'].index(x))
equipments = sorted(set(equipments), key=lambda x: (x[0], int(x[1:])))
raw_cost = {}
sell_price = {}
for rec in records:
    v = rec['values']
    if v['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)':
        for p in products:
            raw_cost[p] = float(v[p])
    if v['Equipment / Cost'] == 'Unit Price (yuan/unit)':
        for p in products:
            sell_price[p] = float(v[p])
x_keys = sorted(proc_time.keys())
y_keys = products
m = gp.Model('factory_opt')
x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(y_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
for eq in equipments:
    terms = []
    for p in products:
        if (eq, p) in proc_time:
            terms.append(proc_time[eq, p] * x[eq, p])
    if terms:
        m.addConstr(gp.quicksum(terms) <= avail_time[eq])
A_eq = [eq for eq in equipments if eq.startswith('A')]
B_eq = [eq for eq in equipments if eq.startswith('B')]
A_I = [eq for eq in A_eq if (eq, 'Product I') in proc_time]
B_I = [eq for eq in B_eq if (eq, 'Product I') in proc_time]
m.addConstr(gp.quicksum((x[eq, 'Product I'] for eq in A_I)) == y['Product I'])
m.addConstr(gp.quicksum((x[eq, 'Product I'] for eq in B_I)) == y['Product I'])
A_II = [eq for eq in A_eq if (eq, 'Product II') in proc_time]
B_II = [eq for eq in B_eq if (eq, 'Product II') in proc_time]
m.addConstr(gp.quicksum((x[eq, 'Product II'] for eq in A_II)) == y['Product II'])
m.addConstr(gp.quicksum((x[eq, 'Product II'] for eq in B_II)) == y['Product II'])
A_III = [eq for eq in A_eq if (eq, 'Product III') in proc_time]
B_III = [eq for eq in B_eq if (eq, 'Product III') in proc_time]
m.addConstr(gp.quicksum((x[eq, 'Product III'] for eq in A_III)) == y['Product III'])
m.addConstr(gp.quicksum((x[eq, 'Product III'] for eq in B_III)) == y['Product III'])
revenue = gp.quicksum((sell_price[p] * y[p] for p in products))
rawmat = gp.quicksum((raw_cost[p] * y[p] for p in products))
equipcost_terms = []
for eq in equipments:
    numer = gp.LinExpr()
    for p in products:
        if (eq, p) in proc_time:
            numer += proc_time[eq, p] * x[eq, p]
    equipcost_terms.append(equip_cost[eq] * numer / avail_time[eq])
equipcost = gp.quicksum(equipcost_terms)
m.setObjective(revenue - rawmat - equipcost, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')