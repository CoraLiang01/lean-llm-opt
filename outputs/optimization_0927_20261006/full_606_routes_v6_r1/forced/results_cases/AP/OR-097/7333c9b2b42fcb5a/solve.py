import gurobipy as gp
from gurobipy import GRB
options = [f'Opt_{i}' for i in range(1, 121)]
assets = [f'Asset_{j}' for j in range(1, 7)]
option_data = {'Opt_1': {'Cost': 9, 'Delta': -0.54, 'Gamma': 0.12, 'Vega': 0.1, 'MaxLong': 9, 'MaxShort': -14}, 'Opt_2': {'Cost': 6, 'Delta': 0.51, 'Gamma': 0.1, 'Vega': 0.16, 'MaxLong': 9, 'MaxShort': -14}, 'Opt_3': {'Cost': 13, 'Delta': 0.17, 'Gamma': 0.02, 'Vega': 0.19, 'MaxLong': 10, 'MaxShort': -7}, 'Opt_4': {'Cost': 10, 'Delta': -0.24, 'Gamma': 0.03, 'Vega': 0.18, 'MaxLong': 7, 'MaxShort': -5}, 'Opt_5': {'Cost': 7, 'Delta': -0.61, 'Gamma': 0.14, 'Vega': 0.11, 'MaxLong': 12, 'MaxShort': -9}, 'Opt_6': {'Cost': 9, 'Delta': -0.26, 'Gamma': 0.09, 'Vega': 0.24, 'MaxLong': 5, 'MaxShort': -13}, 'Opt_7': {'Cost': 12, 'Delta': -0.24, 'Gamma': 0.01, 'Vega': 0.2, 'MaxLong': 10, 'MaxShort': -5}, 'Opt_8': {'Cost': 5, 'Delta': 0.32, 'Gamma': 0.02, 'Vega': 0.16, 'MaxLong': 8, 'MaxShort': -7}, 'Opt_9': {'Cost': 9, 'Delta': 0.19, 'Gamma': 0.1, 'Vega': 0.17, 'MaxLong': 5, 'MaxShort': -8}, 'Opt_10': {'Cost': 13, 'Delta': 0.54, 'Gamma': 0.01, 'Vega': 0.13, 'MaxLong': 11, 'MaxShort': -5}, 'Opt_120': {'Cost': 12, 'Delta': 0.54, 'Gamma': 0.03, 'Vega': 0.19, 'MaxLong': 9, 'MaxShort': -7}}
asset_ref = {'Opt_1': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 1, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_2': {'Asset_1': 1, 'Asset_2': 1, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 0}, 'Opt_3': {'Asset_1': 0, 'Asset_2': 0, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 0, 'Asset_6': 1}, 'Opt_120': {'Asset_1': 0, 'Asset_2': 1, 'Asset_3': 0, 'Asset_4': 0, 'Asset_5': 1, 'Asset_6': 0}}
initial_greeks = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerances = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
for opt in options:
    if opt not in option_data:
        raise ValueError(f'Missing option characteristics for {opt}')
    if opt not in asset_ref:
        raise ValueError(f'Missing asset reference for {opt}')
    for asset in assets:
        if asset not in asset_ref[opt]:
            raise ValueError(f'Missing asset reference for {opt}, {asset}')
m = gp.Model('Option_Hedging')
x_vars = m.addVars(options, vtype=GRB.INTEGER, name='')
z_vars = m.addVars(options, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((option_data[opt]['Cost'] * z_vars[opt] for opt in options)), GRB.MINIMIZE)
for opt in options:
    m.addConstr(z_vars[opt] >= x_vars[opt], name=f'abs1_{opt}')
    m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'abs2_{opt}')
for opt in options:
    m.addConstr(x_vars[opt] >= option_data[opt]['MaxShort'], name=f'short_{opt}')
    m.addConstr(x_vars[opt] <= option_data[opt]['MaxLong'], name=f'long_{opt}')
for greek in ['Delta', 'Gamma', 'Vega']:
    greek_init = initial_greeks[greek]
    tol = tolerances[greek]
    for asset in assets:
        expr = greek_init + gp.quicksum((option_data[opt][greek] * asset_ref[opt][asset] * x_vars[opt] for opt in options))
        m.addConstr(expr <= tol, name=f'{greek}_ub_{asset}')
        m.addConstr(expr >= -tol, name=f'{greek}_lb_{asset}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')