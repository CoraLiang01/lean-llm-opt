import gurobipy as gp
from gurobipy import GRB
models = ['HiFi1', 'HiFi2', 'HiFi3', 'HiFi4', 'HiFi5', 'HiFi6', 'HiFi7', 'HiFi8', 'HiFi9', 'HiFi10', 'HiFi11', 'HiFi12', 'HiFi13', 'HiFi14', 'HiFi15', 'HiFi16', 'HiFi17', 'HiFi18', 'HiFi19', 'HiFi20', 'HiFi21', 'HiFi22', 'HiFi23', 'HiFi24', 'HiFi25', 'HiFi26', 'HiFi27', 'HiFi28', 'HiFi29', 'HiFi30', 'HiFi31', 'HiFi32', 'HiFi33', 'HiFi34', 'HiFi35', 'HiFi36', 'HiFi37', 'HiFi38', 'HiFi39', 'HiFi40', 'HiFi41', 'HiFi42', 'HiFi43', 'HiFi44', 'HiFi45', 'HiFi46', 'HiFi47', 'HiFi48', 'HiFi49', 'HiFi50', 'HiFi51', 'HiFi52', 'HiFi53', 'HiFi54', 'HiFi55', 'HiFi56', 'HiFi57', 'HiFi58', 'HiFi59', 'HiFi60', 'HiFi61', 'HiFi62', 'HiFi63', 'HiFi64', 'HiFi65', 'HiFi66', 'HiFi67', 'HiFi68', 'HiFi69', 'HiFi70', 'HiFi71', 'HiFi72', 'HiFi73', 'HiFi74', 'HiFi75', 'HiFi76', 'HiFi77', 'HiFi78', 'HiFi79', 'HiFi80', 'HiFi81', 'HiFi82', 'HiFi83', 'HiFi84', 'HiFi85', 'HiFi86', 'HiFi87', 'HiFi88', 'HiFi89', 'HiFi90', 'HiFi91', 'HiFi92', 'HiFi93', 'HiFi94', 'HiFi95', 'HiFi96', 'HiFi97', 'HiFi98', 'HiFi99', 'HiFi100', 'HiFi101']
workstations = ['W1', 'W2', 'W3']
capacities = {'W1': 1296, 'W2': 1238.4, 'W3': 1267.2}
a_1 = [6, 4, 6, 7, 6, 6, 8, 9, 6, 7, 1, 2, 4, 7, 3, 8, 3, 2, 4, 5, 8, 3, 2, 3, 9, 7, 3, 5, 7, 6, 2, 1, 5, 6, 5, 1, 7, 9, 8, 3, 3, 8, 2, 3, 3, 8, 9, 2, 3, 4, 2, 9, 2, 1, 8, 8, 4, 4, 6, 1, 6, 5, 3, 5, 1, 6, 6, 5, 3, 4, 3, 8, 1, 2, 3, 2, 8, 4, 4, 2, 7, 5, 1, 6, 4, 1, 3, 8, 3, 3, 3, 3, 6, 7, 6, 2, 1, 8, 9, 7, 9]
a_2 = [5, 5, 5, 1, 7, 8, 7, 5, 6, 8, 9, 9, 2, 6, 9, 4, 1, 2, 9, 3, 8, 5, 9, 5, 8, 7, 1, 1, 9, 7, 1, 9, 6, 4, 7, 4, 8, 6, 5, 3, 6, 7, 6, 2, 1, 1, 3, 8, 4, 3, 6, 9, 8, 7, 2, 2, 5, 4, 3, 8, 8, 6, 6, 3, 1, 6, 2, 6, 1, 3, 7, 1, 1, 2, 8, 7, 8, 8, 7, 5, 2, 5, 6, 2, 3, 2, 3, 8, 4, 9, 6, 1, 4, 8, 8, 6, 8, 5, 5, 8, 3]
a_3 = [4, 6, 5, 2, 6, 5, 3, 3, 4, 8, 6, 3, 3, 3, 7, 8, 3, 8, 1, 5, 3, 8, 5, 8, 4, 8, 6, 7, 9, 5, 3, 6, 3, 3, 3, 8, 4, 6, 3, 8, 3, 7, 5, 3, 1, 8, 9, 6, 6, 4, 7, 1, 9, 9, 3, 9, 6, 5, 7, 8, 9, 9, 8, 5, 4, 4, 3, 3, 8, 8, 2, 4, 9, 6, 7, 6, 7, 3, 1, 7, 6, 4, 3, 5, 7, 6, 3, 5, 2, 2, 9, 3, 6, 9, 7, 2, 4, 5, 8, 1, 6]
a = {'W1': dict(zip(models, a_1)), 'W2': dict(zip(models, a_2)), 'W3': dict(zip(models, a_3))}
for w in workstations:
    if set(a[w].keys()) != set(models):
        raise ValueError(f'Missing coefficients for {w}')
m = gp.Model('radio_idle_time_min')
x_vars = m.addVars(models, lb=0, vtype=GRB.INTEGER, name='')
idle_exprs = {}
for w in workstations:
    idle_exprs[w] = capacities[w] - gp.quicksum((a[w][model] * x_vars[model] for model in models))
m.setObjective(gp.quicksum((idle_exprs[w] for w in workstations)), GRB.MINIMIZE)
for w in workstations:
    m.addConstr(gp.quicksum((a[w][model] * x_vars[model] for model in models)) <= capacities[w], name=f'cap_{w}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')