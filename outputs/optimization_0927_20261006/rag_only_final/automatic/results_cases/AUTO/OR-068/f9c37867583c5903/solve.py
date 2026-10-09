from gurobipy import Model, GRB, quicksum
managers = ['MA', 'MB', 'MC', 'MD', 'ME', 'MF']
projects = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
cost_matrix = {'MA': {'P1': 2216, 'P2': 1911, 'P3': 1661, 'P4': 2122, 'P5': 1442, 'P6': 1442}, 'MB': {'P1': 1100, 'P2': 1271, 'P3': 2764, 'P4': 2557, 'P5': 1036, 'P6': 1036}, 'MC': {'P1': 2827, 'P2': 2784, 'P3': 2206, 'P4': 2216, 'P5': 2677, 'P6': 2677}, 'MD': {'P1': 2627, 'P2': 1273, 'P3': 2610, 'P4': 1957, 'P5': 1594, 'P6': 1594}, 'ME': {'P1': 3359, 'P2': 1003, 'P3': 2554, 'P4': 1706, 'P5': 2065, 'P6': 2065}, 'MF': {'P1': 1579, 'P2': 2289, 'P3': 2368, 'P4': 1922, 'P5': 2740, 'P6': 2740}}
if set(cost_matrix.keys()) != set(managers):
    raise ValueError('Manager keys in cost_matrix do not match managers set.')
for m in managers:
    if set(cost_matrix[m].keys()) != set(projects):
        raise ValueError(f'Project keys in cost_matrix for manager {m} do not match projects set.')

def build_assignment_model():
    model = Model()
    model.Params.MIPGap = 0.0001
    x_vars = model.addVars(managers, projects, vtype=GRB.BINARY, name='')
    manager_constrs = model.addConstrs((quicksum((x_vars[m, p] for p in projects)) == 1 for m in managers), name='')
    project_constrs = model.addConstrs((quicksum((x_vars[m, p] for m in managers)) == 1 for p in projects), name='')
    model.setObjective(quicksum((cost_matrix[m][p] * x_vars[m, p] for m in managers for p in projects)), GRB.MINIMIZE)
    return model
m = build_assignment_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')