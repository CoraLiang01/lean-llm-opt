import gurobipy as gp
from gurobipy import GRB
options = [f'Opt_{i}' for i in range(1, 121)]
assets = [f'Asset_{j}' for j in range(1, 7)]
Cost = {'Opt_1': 9, 'Opt_2': 6, 'Opt_3': 13, 'Opt_4': 10, 'Opt_5': 7, 'Opt_6': 9, 'Opt_7': 12, 'Opt_8': 5, 'Opt_9': 9, 'Opt_10': 13, 'Opt_120': 12}
Delta = {'Opt_1': -0.54, 'Opt_2': 0.51, 'Opt_3': 0.17, 'Opt_4': -0.24, 'Opt_5': -0.61, 'Opt_6': -0.26, 'Opt_7': -0.24, 'Opt_8': 0.32, 'Opt_9': 0.19, 'Opt_10': 0.54, 'Opt_120': 0.54}
Gamma = {'Opt_1': 0.12, 'Opt_2': 0.1, 'Opt_3': 0.02, 'Opt_4': 0.03, 'Opt_5': 0.14, 'Opt_6': 0.09, 'Opt_7': 0.01, 'Opt_8': 0.02, 'Opt_9': 0.1, 'Opt_10': 0.01, 'Opt_120': 0.03}
Vega = {'Opt_1': 0.1, 'Opt_2': 0.16, 'Opt_3': 0.19, 'Opt_4': 0.18, 'Opt_5': 0.11, 'Opt_6': 0.24, 'Opt_7': 0.2, 'Opt_8': 0.16, 'Opt_9': 0.17, 'Opt_10': 0.13, 'Opt_120': 0.19}
MaxLong = {'Opt_1': 9, 'Opt_2': 9, 'Opt_3': 10, 'Opt_4': 7, 'Opt_5': 12, 'Opt_6': 5, 'Opt_7': 10, 'Opt_8': 8, 'Opt_9': 5, 'Opt_10': 11, 'Opt_120': 9}
MaxShort = {'Opt_1': -14, 'Opt_2': -14, 'Opt_3': -7, 'Opt_4': -5, 'Opt_5': -9, 'Opt_6': -13, 'Opt_7': -5, 'Opt_8': -7, 'Opt_9': -8, 'Opt_10': -5, 'Opt_120': -7}
A = {'Opt_1': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 1, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_2': {'Asset_1': 1, 'Asset_2': 1, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_3': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 1}, 'Opt_4': {'Asset_1': 0, 'Asset_2': 1, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 1, 'Asset_6': 0}, 'Opt_5': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 1, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_6': {'Asset_1': 1, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 1, 'Asset_6': 0}, 'Opt_7': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 1, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 1}, 'Opt_8': {'Asset_1': 0, 'Asset_2': 1, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_9': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 1, 'Asset_5': 1, 'Asset_6': 0}, 'Opt_10': {'Asset_1': 1, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 1}, 'Opt_120': {'Asset_1': 0, 'Asset_2': 1, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 1, 'Asset_6': 0}}
for opt in options:
    if opt not in Cost or opt not in Delta or opt not in Gamma or (opt not in Vega) or (opt not in MaxLong) or (opt not in MaxShort) or (opt not in A):
        raise ValueError(f'Missing data for {opt}')
    for asset in assets:
        if asset not in A[opt]:
            raise ValueError(f'Missing asset reference for {opt}, {asset}')
Delta_initial = 0.25
Gamma_initial = 0.08
Vega_initial = 0.17
Tolerance_Delta = 0.06
Tolerance_Gamma = 0.05
Tolerance_Vega = 0.07
m = gp.Model('OptionHedging')
x_vars = m.addVars(options, vtype=GRB.INTEGER, name='')
y_vars = m.addVars(options, lb=0, vtype=GRB.CONTINUOUS, name='')
for opt in options:
    m.addConstr(x_vars[opt] >= MaxShort[opt], name=f'lb_{opt}')
    m.addConstr(x_vars[opt] <= MaxLong[opt], name=f'ub_{opt}')
for opt in options:
    m.addConstr(y_vars[opt] >= x_vars[opt], name=f'y_ge_x_{opt}')
    m.addConstr(y_vars[opt] >= -x_vars[opt], name=f'y_ge_negx_{opt}')
    m.addConstr(y_vars[opt] >= 0, name=f'y_ge_0_{opt}')
m.setObjective(gp.quicksum((Cost[opt] * y_vars[opt] for opt in options)), GRB.MINIMIZE)
delta_expr = Delta_initial + gp.quicksum((Delta[opt] * A[opt][asset] * x_vars[opt] for opt in options for asset in assets))
m.addConstr(delta_expr <= Tolerance_Delta, name='Delta_upper')
m.addConstr(delta_expr >= -Tolerance_Delta, name='Delta_lower')
gamma_expr = Gamma_initial + gp.quicksum((Gamma[opt] * A[opt][asset] * x_vars[opt] for opt in options for asset in assets))
m.addConstr(gamma_expr <= Tolerance_Gamma, name='Gamma_upper')
m.addConstr(gamma_expr >= -Tolerance_Gamma, name='Gamma_lower')
vega_expr = Vega_initial + gp.quicksum((Vega[opt] * A[opt][asset] * x_vars[opt] for opt in options for asset in assets))
m.addConstr(vega_expr <= Tolerance_Vega, name='Vega_upper')
m.addConstr(vega_expr >= -Tolerance_Vega, name='Vega_lower')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')