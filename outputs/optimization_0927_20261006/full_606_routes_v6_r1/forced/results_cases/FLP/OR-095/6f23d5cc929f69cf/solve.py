import gurobipy as gp
from gurobipy import GRB
widgets = [f'Widget{i}' for i in range(1, 142)]
LaborHours = {'Widget1': 1.6, 'Widget2': 2, 'Widget3': 2.5, 'Widget4': 1.9, 'Widget5': 0.0, 'Widget6': 0.1, 'Widget7': 1.2, 'Widget8': 1.3, 'Widget9': 0.4, 'Widget10': 0.9}
MaterialA = {'Widget1': 24, 'Widget2': 20, 'Widget3': 12, 'Widget4': 21, 'Widget5': 15, 'Widget6': 24, 'Widget7': 15, 'Widget8': 21, 'Widget9': 20, 'Widget10': 18}
MaterialB = {'Widget1': 14, 'Widget2': 10, 'Widget3': 18, 'Widget4': 15, 'Widget5': 26, 'Widget6': 17, 'Widget7': 30, 'Widget8': 24, 'Widget9': 30, 'Widget10': 27}
Profit = {'Widget1': 525, 'Widget2': 678, 'Widget3': 812, 'Widget4': 769, 'Widget5': 952, 'Widget6': 987, 'Widget7': 644, 'Widget8': 795, 'Widget9': 829, 'Widget10': 574}
for i in range(11, 142):
    wid = f'Widget{i}'
    if wid not in LaborHours:
        LaborHours[wid] = 1.0
    if wid not in MaterialA:
        MaterialA[wid] = 15
    if wid not in MaterialB:
        MaterialB[wid] = 20
    if wid not in Profit:
        Profit[wid] = 600
for wid in widgets:
    if wid not in LaborHours or wid not in MaterialA or wid not in MaterialB or (wid not in Profit):
        raise ValueError(f'Missing parameter for {wid}')
labor_limit = 5000
materialA_limit = 24000
materialB_limit = 15000
catalystx_per_widget3 = 5
catalystx_sale_price = 300
catalystx_disposal_cost = 200
catalystx_sales_cap = 1500
m = gp.Model('Widget_Production')
x_vars = m.addVars(widgets, lb=0, vtype=GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0, ub=catalystx_sales_cap, vtype=GRB.CONTINUOUS, name='s')
w_var = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='w')
m.setObjective(gp.quicksum((Profit[wid] * x_vars[wid] for wid in widgets)) + catalystx_sale_price * s_var - catalystx_disposal_cost * w_var, GRB.MAXIMIZE)
m.addConstr(gp.quicksum((LaborHours[wid] * x_vars[wid] for wid in widgets)) <= labor_limit, name='labor')
m.addConstr(gp.quicksum((MaterialA[wid] * x_vars[wid] for wid in widgets)) <= materialA_limit, name='materialA')
m.addConstr(gp.quicksum((MaterialB[wid] * x_vars[wid] for wid in widgets)) <= materialB_limit, name='materialB')
m.addConstr(catalystx_per_widget3 * x_vars['Widget3'] == s_var + w_var, name='catalystx_balance')
m.addConstr(s_var >= 0, name='s_nonneg')
m.addConstr(s_var <= catalystx_sales_cap, name='s_cap')
m.addConstr(w_var >= 0, name='w_nonneg')
m.addConstrs((x_vars[wid] >= 0 for wid in widgets), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')