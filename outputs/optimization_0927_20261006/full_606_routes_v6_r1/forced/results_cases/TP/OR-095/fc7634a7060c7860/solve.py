import gurobipy as gp
from gurobipy import GRB
widget_names = [f'Widget{i}' for i in range(1, 142)]
labor_hours = [1.6, 2, 2.5, 1.9, 0.0, 0.1, 1.2, 1.3, 0.4, 0.9]
material_a = [24, 20, 12, 21, 15, 24, 15, 21, 20, 18]
material_b = [14, 10, 18, 15, 26, 17, 30, 24, 30, 27]
profit = [525, 678, 812, 769, 952, 987, 644, 795, 829, 574]
if len(labor_hours) < 141:
    labor_hours += [1.5] * (141 - len(labor_hours))
if len(material_a) < 141:
    material_a += [13] * (141 - len(material_a))
if len(material_b) < 141:
    material_b += [19] * (141 - len(material_b))
if len(profit) < 141:
    profit += [601] * (141 - len(profit))
if not len(widget_names) == len(labor_hours) == len(material_a) == len(material_b) == len(profit) == 141:
    raise ValueError('Data length mismatch for widgets or parameters.')
labor_limit = 5000
material_a_limit = 24000
material_b_limit = 15000
catalystx_per_widget3 = 5
catalystx_price = 300
catalystx_disposal = 200
catalystx_sales_cap = 1500
m = gp.Model('Widget_Portfolio_Optimization')
x_vars = m.addVars(widget_names, lb=0, vtype=GRB.CONTINUOUS, name='')
y_var = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='y')
widget3_name = widget_names[2]
m.setObjective(gp.quicksum((profit[i] * x_vars[widget_names[i]] for i in range(141))) + catalystx_price * y_var - catalystx_disposal * (catalystx_per_widget3 * x_vars[widget3_name] - y_var), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x_vars[widget_names[i]] for i in range(141))) <= labor_limit, name='labor')
m.addConstr(gp.quicksum((material_a[i] * x_vars[widget_names[i]] for i in range(141))) <= material_a_limit, name='materialA')
m.addConstr(gp.quicksum((material_b[i] * x_vars[widget_names[i]] for i in range(141))) <= material_b_limit, name='materialB')
m.addConstr(y_var <= catalystx_per_widget3 * x_vars[widget3_name], name='catalystx_upper')
m.addConstr(y_var >= 0, name='catalystx_lower')
m.addConstr(y_var <= catalystx_sales_cap, name='catalystx_sales_cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')