import gurobipy as gp
from gurobipy import GRB
options = [f'Option_{i + 1}' for i in range(120)]
assets = [f'Asset_{j + 1}' for j in range(6)]
Cost = {f'Option_{i + 1}': 1.0 + 0.01 * i for i in range(120)}
Delta = {f'Option_{i + 1}': -0.5 + i * 0.01 for i in range(120)}
Gamma = {f'Option_{i + 1}': 0.01 * (i % 10 - 5) for i in range(120)}
Vega = {f'Option_{i + 1}': 0.02 * (i % 7 - 3) for i in range(120)}
MaxLong = {f'Option_{i + 1}': 10 + i % 5 for i in range(120)}
MaxShort = {f'Option_{i + 1}': -5 - i % 3 for i in range(120)}
A = {}
for i, opt in enumerate(options):
    for j, asset in enumerate(assets):
        if (i + j) % 6 == 0 or (i + j) % 7 == 0:
            A[opt, asset] = 1
        else:
            A[opt, asset] = 0
Delta_initial = 0.25
Gamma_initial = 0.08
Vega_initial = 0.17
Tolerance_Delta = 0.06
Tolerance_Gamma = 0.05
Tolerance_Vega = 0.07
for opt in options:
    if opt not in Cost or opt not in Delta or opt not in Gamma or (opt not in Vega) or (opt not in MaxLong) or (opt not in MaxShort):
        raise ValueError(f'Missing data for option {opt}')
for opt in options:
    for asset in assets:
        if (opt, asset) not in A:
            raise ValueError(f'Missing asset reference for ({opt}, {asset})')
m = gp.Model('OptionHedging')
x = m.addVars(options, lb={opt: MaxShort[opt] for opt in options}, ub={opt: MaxLong[opt] for opt in options}, vtype=GRB.INTEGER, name='')
y = m.addVars(options, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((Cost[opt] * y[opt] for opt in options)), GRB.MINIMIZE)
for opt in options:
    m.addConstr(y[opt] >= x[opt], name=f'abs1_{opt}')
    m.addConstr(y[opt] >= -x[opt], name=f'abs2_{opt}')
    m.addConstr(y[opt] >= 0, name=f'abs3_{opt}')
delta_expr = Delta_initial + gp.quicksum((Delta[opt] * A[opt, asset] * x[opt] for opt in options for asset in assets))
m.addConstr(delta_expr <= Tolerance_Delta, name='delta_upper')
m.addConstr(delta_expr >= -Tolerance_Delta, name='delta_lower')
gamma_expr = Gamma_initial + gp.quicksum((Gamma[opt] * A[opt, asset] * x[opt] for opt in options for asset in assets))
m.addConstr(gamma_expr <= Tolerance_Gamma, name='gamma_upper')
m.addConstr(gamma_expr >= -Tolerance_Gamma, name='gamma_lower')
vega_expr = Vega_initial + gp.quicksum((Vega[opt] * A[opt, asset] * x[opt] for opt in options for asset in assets))
m.addConstr(vega_expr <= Tolerance_Vega, name='vega_upper')
m.addConstr(vega_expr >= -Tolerance_Vega, name='vega_lower')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')