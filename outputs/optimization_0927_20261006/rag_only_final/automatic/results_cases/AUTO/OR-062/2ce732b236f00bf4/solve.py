from gurobipy import Model, GRB, quicksum
F = ['MOUNT AYR', 'WAUKEE', 'WAVERLY', 'PELLA', 'DES MOINES']
S = ['CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']
f = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
d = {'CLARINDA': 2397, 'FORT MADISON': 1889, 'SIOUX CITY': 2518, 'TOLEDO': 3218, 'BANCROFT': 1813}
c = {'MOUNT AYR': {'CLARINDA': 694.68, 'FORT MADISON': 17.48, 'SIOUX CITY': 20.07, 'TOLEDO': 199.02, 'BANCROFT': 1685.53}, 'WAUKEE': {'CLARINDA': 15.13, 'FORT MADISON': 1.5, 'SIOUX CITY': 1.43, 'TOLEDO': 27.88, 'BANCROFT': 90.69}, 'WAVERLY': {'CLARINDA': 2.34, 'FORT MADISON': 349.34, 'SIOUX CITY': 246.6, 'TOLEDO': 41.3, 'BANCROFT': 78.73}, 'PELLA': {'CLARINDA': 1181.6, 'FORT MADISON': 1458.53, 'SIOUX CITY': 1646.36, 'TOLEDO': 1924.55, 'BANCROFT': 38.93}, 'DES MOINES': {'CLARINDA': 1030.8, 'FORT MADISON': 43.48, 'SIOUX CITY': 932.43, 'TOLEDO': 55.39, 'BANCROFT': 103.84}}
if set(f.keys()) != set(F):
    raise ValueError('Fixed cost keys do not match supplier set F.')
if set(d.keys()) != set(S):
    raise ValueError('Demand keys do not match store set S.')
if set(c.keys()) != set(F):
    raise ValueError('Transportation cost row keys do not match supplier set F.')
for i in F:
    if set(c[i].keys()) != set(S):
        raise ValueError(f'Transportation cost column keys for supplier {i} do not match store set S.')
m = Model()
m.Params.MIPGap = 0.0001
y_vars = m.addVars(F, vtype=GRB.BINARY, lb=0, ub=1, name='')
x_vars = m.addVars(F, S, vtype=GRB.CONTINUOUS, lb=0, name='')
for j in S:
    m.addConstr(quicksum((x_vars[i, j] for i in F)) == d[j], name='demand_%s' % j)
for i in F:
    for j in S:
        m.addConstr(x_vars[i, j] <= d[j] * y_vars[i], name='link_%s_%s' % (i, j))
m.setObjective(quicksum((f[i] * y_vars[i] for i in F)) + quicksum((c[i][j] * x_vars[i, j] for i in F for j in S)), GRB.MINIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver status:', m.Status)