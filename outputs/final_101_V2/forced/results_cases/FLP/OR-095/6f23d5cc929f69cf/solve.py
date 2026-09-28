import gurobipy as gp
from gurobipy import GRB
widgets = [f'Widget{i}' for i in range(1, 142)]
l_i = {'Widget1': 1.6, 'Widget2': 2.0, 'Widget3': 2.5, 'Widget4': 1.9, 'Widget141': 1.2}
a_i = {'Widget1': 24, 'Widget2': 20, 'Widget3': 12, 'Widget4': 21, 'Widget141': 11}
b_i = {'Widget1': 14, 'Widget2': 10, 'Widget3': 18, 'Widget4': 15, 'Widget141': 16}
p_i = {'Widget1': 525, 'Widget2': 678, 'Widget3': 812, 'Widget4': 769, 'Widget141': 593}
for i in range(5, 141):
    wid = f'Widget{i}'
    if wid not in l_i:
        l_i[wid] = 1.5
        a_i[wid] = 15
        b_i[wid] = 12
        p_i[wid] = 600
L = 5000
A = 24000
B = 15000
catalystx_sale_price = 300
catalystx_disposal_cost = 200
catalystx_generation_per_unit = 5
catalystx_sales_cap = 1500
if set(widgets) != set(l_i) or set(widgets) != set(a_i) or set(widgets) != set(b_i) or (set(widgets) != set(p_i)):
    raise ValueError('Missing data for some widgets in l_i, a_i, b_i, or p_i.')
m = gp.Model('Widget_Production_Optimization')
x = m.addVars(widgets, lb=0, vtype=GRB.CONTINUOUS, name='')
s = m.addVar(lb=0, ub=catalystx_sales_cap, vtype=GRB.CONTINUOUS, name='s')
w = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='w')
m.setObjective(gp.quicksum((p_i[i] * x[i] for i in widgets)) + catalystx_sale_price * s - catalystx_disposal_cost * w, GRB.MAXIMIZE)
m.addConstr(gp.quicksum((l_i[i] * x[i] for i in widgets)) <= L, name='labor')
m.addConstr(gp.quicksum((a_i[i] * x[i] for i in widgets)) <= A, name='matA')
m.addConstr(gp.quicksum((b_i[i] * x[i] for i in widgets)) <= B, name='matB')
m.addConstr(catalystx_generation_per_unit * x['Widget3'] == s + w, name='catalystx_balance')
m.addConstr(s >= 0, name='s_nonneg')
m.addConstr(s <= catalystx_sales_cap, name='s_cap')
m.addConstr(w >= 0, name='w_nonneg')
m.addConstrs((x[i] >= 0 for i in widgets), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')