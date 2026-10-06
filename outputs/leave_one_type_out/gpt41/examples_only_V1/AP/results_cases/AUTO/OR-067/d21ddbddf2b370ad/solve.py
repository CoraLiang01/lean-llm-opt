from gurobipy import Model, GRB

def solve_assignment_problem():
    managers = ['MA', 'MB', 'MC']
    projects = ['P1', 'P2', 'P3']
    c = {'MA': {'P1': 3000, 'P2': 3200, 'P3': 3100}, 'MB': {'P1': 2800, 'P2': 3300, 'P3': 2900}, 'MC': {'P1': 2900, 'P2': 3100, 'P3': 3000}}
    if set(c.keys()) != set(managers):
        raise ValueError('Cost matrix manager keys do not match managers set.')
    for m in managers:
        if set(c[m].keys()) != set(projects):
            raise ValueError(f'Cost matrix for manager {m} does not cover all projects.')
    model = Model()
    model.Params.MIPGap = 0.0001
    x = model.addVars(managers, projects, vtype=GRB.BINARY, lb=0, ub=1, name='')
    model.setObjective(sum((c[m][p] * x[m, p] for m in managers for p in projects)), GRB.MINIMIZE)
    for m in managers:
        model.addConstr(sum((x[m, p] for p in projects)) == 1, name='')
    for p in projects:
        model.addConstr(sum((x[m, p] for m in managers)) == 1, name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal {model.ObjVal}')
        for m in managers:
            for p in projects:
                var = x[m, p]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_assignment_problem()