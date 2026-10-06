LEGACY_OBSERVATION = 'Equipment / Cost,Product I,Product II,Product III,Available Equipment Operating Time,Equipment Cost at Full Load (yuan)\nA1,5,10,,6000,300\nA2,7,9,12,10000,321\nA3,6,11,2,8000,203\nB1,6,8,,4000,250\nB2,4,,11,7000,783\nB3,7,,,4000,200\nB4,3,5,8,5000,300\nRaw Material Cost (yuan/unit),0.25,0.35,0.5,,\nUnit Price (yuan/unit),1.25,2,2.8,,'
LEGACY_RECORDS = [{'source': '', 'values': {'Equipment / Cost': 'A1', 'Product I': '5', 'Product II': '10', 'Product III': '', 'Available Equipment Operating Time': '6000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '', 'values': {'Equipment / Cost': 'A2', 'Product I': '7', 'Product II': '9', 'Product III': '12', 'Available Equipment Operating Time': '10000', 'Equipment Cost at Full Load (yuan)': '321'}}, {'source': '', 'values': {'Equipment / Cost': 'A3', 'Product I': '6', 'Product II': '11', 'Product III': '2', 'Available Equipment Operating Time': '8000', 'Equipment Cost at Full Load (yuan)': '203'}}, {'source': '', 'values': {'Equipment / Cost': 'B1', 'Product I': '6', 'Product II': '8', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '250'}}, {'source': '', 'values': {'Equipment / Cost': 'B2', 'Product I': '4', 'Product II': '', 'Product III': '11', 'Available Equipment Operating Time': '7000', 'Equipment Cost at Full Load (yuan)': '783'}}, {'source': '', 'values': {'Equipment / Cost': 'B3', 'Product I': '7', 'Product II': '', 'Product III': '', 'Available Equipment Operating Time': '4000', 'Equipment Cost at Full Load (yuan)': '200'}}, {'source': '', 'values': {'Equipment / Cost': 'B4', 'Product I': '3', 'Product II': '5', 'Product III': '8', 'Available Equipment Operating Time': '5000', 'Equipment Cost at Full Load (yuan)': '300'}}, {'source': '', 'values': {'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}, {'source': '', 'values': {'Equipment / Cost': 'Unit Price (yuan/unit)', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8', 'Available Equipment Operating Time': '', 'Equipment Cost at Full Load (yuan)': ''}}]
import gurobipy as gp
from gurobipy import GRB
equipments = []
products = []
c = {}
T = {}
F = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq in ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4']:
        if eq not in equipments:
            equipments.append(eq)
        for prod in ['Product I', 'Product II', 'Product III']:
            if v[prod] != '':
                p = prod.replace('Product ', '')
                if p not in products:
                    products.append(p)
                c[eq, p] = float(v[prod])
        T[eq] = float(v['Available Equipment Operating Time'])
        F[eq] = float(v['Equipment Cost at Full Load (yuan)'])
r = {}
p = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    eq = v['Equipment / Cost']
    if eq == 'Raw Material Cost (yuan/unit)':
        for prod in ['Product I', 'Product II', 'Product III']:
            if v[prod] != '':
                r[prod.replace('Product ', '')] = float(v[prod])
    if eq == 'Unit Price (yuan/unit)':
        for prod in ['Product I', 'Product II', 'Product III']:
            if v[prod] != '':
                p[prod.replace('Product ', '')] = float(v[prod])
allowed = set(c.keys())
m = gp.Model('factory_opt')
x = m.addVars(allowed, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(['I', 'II', 'III'], lb=0, vtype=GRB.CONTINUOUS, name='')
u = m.addVars(equipments, lb=0, vtype=GRB.CONTINUOUS, name='')
A_eq = ['A1', 'A2', 'A3']
B_eq = ['B1', 'B2', 'B3', 'B4']
m.addConstr(y['I'] == gp.quicksum((x[e, 'I'] for e in A_eq if (e, 'I') in allowed)), name='prodI_A')
m.addConstr(y['I'] == gp.quicksum((x[e, 'I'] for e in B_eq if (e, 'I') in allowed)), name='prodI_B')
m.addConstr(y['II'] == gp.quicksum((x[e, 'II'] for e in A_eq if (e, 'II') in allowed)), name='prodII_A')
m.addConstr(y['II'] == gp.quicksum((x[e, 'II'] for e in B_eq if (e, 'II') in allowed)), name='prodII_B')
A_eq_III = ['A2', 'A3']
B_eq_III = ['B2', 'B4']
m.addConstr(y['III'] == gp.quicksum((x[e, 'III'] for e in A_eq_III if (e, 'III') in allowed)), name='prodIII_A')
m.addConstr(y['III'] == gp.quicksum((x[e, 'III'] for e in B_eq_III if (e, 'III') in allowed)), name='prodIII_B')
for e in equipments:
    expr = gp.LinExpr()
    for prod in products:
        if (e, prod) in allowed:
            expr += c[e, prod] * x[e, prod]
    m.addConstr(expr <= T[e], name=f'time_{e}')
    m.addConstr(u[e] == expr, name=f'u_{e}')
profit_expr = p['I'] * y['I'] + p['II'] * y['II'] + p['III'] * y['III'] - r['I'] * y['I'] - r['II'] * y['II'] - r['III'] * y['III'] - gp.quicksum((F[e] * u[e] / T[e] for e in equipments))
m.setObjective(profit_expr, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')