import gurobipy as gp
from gurobipy import GRB
options = [f'Opt_{i + 1}' for i in range(120)]
assets = [f'Asset_{j + 1}' for j in range(6)]
option_data = {'Opt_1': {'Cost': 9, 'Delta': -0.54, 'Gamma': 0.12, 'Vega': 0.1, 'MaxLong': 9, 'MaxShort': -14}, 'Opt_2': {'Cost': 6, 'Delta': 0.51, 'Gamma': 0.1, 'Vega': 0.16, 'MaxLong': 9, 'MaxShort': -14}, 'Opt_3': {'Cost': 13, 'Delta': 0.17, 'Gamma': 0.02, 'Vega': 0.19, 'MaxLong': 10, 'MaxShort': -7}, 'Opt_4': {'Cost': 10, 'Delta': -0.24, 'Gamma': 0.03, 'Vega': 0.18, 'MaxLong': 7, 'MaxShort': -5}, 'Opt_5': {'Cost': 7, 'Delta': -0.61, 'Gamma': 0.14, 'Vega': 0.11, 'MaxLong': 12, 'MaxShort': -9}, 'Opt_6': {'Cost': 9, 'Delta': -0.26, 'Gamma': 0.09, 'Vega': 0.24, 'MaxLong': 5, 'MaxShort': -13}, 'Opt_7': {'Cost': 12, 'Delta': -0.24, 'Gamma': 0.01, 'Vega': 0.2, 'MaxLong': 10, 'MaxShort': -5}, 'Opt_8': {'Cost': 5, 'Delta': 0.32, 'Gamma': 0.02, 'Vega': 0.16, 'MaxLong': 8, 'MaxShort': -7}, 'Opt_9': {'Cost': 9, 'Delta': 0.19, 'Gamma': 0.1, 'Vega': 0.17, 'MaxLong': 5, 'MaxShort': -8}, 'Opt_10': {'Cost': 13, 'Delta': 0.54, 'Gamma': 0.01, 'Vega': 0.13, 'MaxLong': 11, 'MaxShort': -5}}
for i in range(10, 120):
    option_data[f'Opt_{i + 1}'] = {'Cost': 10 + i % 5, 'Delta': -0.5 + 0.01 * (i % 100), 'Gamma': 0.01 + 0.01 * (i % 10), 'Vega': 0.1 + 0.01 * (i % 20), 'MaxLong': 5 + i % 8, 'MaxShort': -14 + i % 10}
A = {'Opt_1': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 1, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_2': {'Asset_1': 1, 'Asset_2': 1, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_3': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 1}, 'Opt_4': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 1, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_5': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 1, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_6': {'Asset_1': 1, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_7': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 1, 'Asset_6': 1}, 'Opt_8': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 1, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_9': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 1, 'Asset_5': 0, 'Asset_6': 1}, 'Opt_10': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 1, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 0}}
for i in range(10, 120):
    A[f'Opt_{i + 1}'] = {f'Asset_{j + 1}': 1 if (i + j) % 6 == 0 else 0 for j in range(6)}
Delta_init = 0.25
Gamma_init = 0.08
Vega_init = 0.17
T_Delta = 0.06
T_Gamma = 0.05
T_Vega = 0.07
for opt in options:
    if opt not in option_data:
        raise ValueError(f'Missing option characteristics for {opt}')
    if opt not in A:
        raise ValueError(f'Missing asset reference for {opt}')
    for asset in assets:
        if asset not in A[opt]:
            raise ValueError(f'Missing asset {asset} for {opt}')
m = gp.Model('Option_Hedging')
x_vars = m.addVars(options, lb={opt: option_data[opt]['MaxShort'] for opt in options}, ub={opt: option_data[opt]['MaxLong'] for opt in options}, vtype=GRB.INTEGER, name='')
z_vars = m.addVars(options, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((option_data[opt]['Cost'] * z_vars[opt] for opt in options)), GRB.MINIMIZE)
for opt in options:
    m.addConstr(z_vars[opt] >= x_vars[opt], name=f'z_ge_x_{opt}')
    m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'z_ge_negx_{opt}')
    m.addConstr(z_vars[opt] >= 0, name=f'z_ge_0_{opt}')
delta_expr = Delta_init + gp.quicksum((option_data[opt]['Delta'] * A[opt][asset] * x_vars[opt] for opt in options for asset in assets))
m.addConstr(delta_expr <= T_Delta, name='Delta_upper')
m.addConstr(delta_expr >= -T_Delta, name='Delta_lower')
gamma_expr = Gamma_init + gp.quicksum((option_data[opt]['Gamma'] * A[opt][asset] * x_vars[opt] for opt in options for asset in assets))
m.addConstr(gamma_expr <= T_Gamma, name='Gamma_upper')
m.addConstr(gamma_expr >= -T_Gamma, name='Gamma_lower')
vega_expr = Vega_init + gp.quicksum((option_data[opt]['Vega'] * A[opt][asset] * x_vars[opt] for opt in options for asset in assets))
m.addConstr(vega_expr <= T_Vega, name='Vega_upper')
m.addConstr(vega_expr >= -T_Vega, name='Vega_lower')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')