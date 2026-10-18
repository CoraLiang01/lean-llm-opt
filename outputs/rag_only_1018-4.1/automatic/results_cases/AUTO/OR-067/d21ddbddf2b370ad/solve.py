LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MA",\n      "P1": "3000",\n      "P2": "3200",\n      "P3": "3100"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MB",\n      "P1": "2800",\n      "P2": "3300",\n      "P3": "2900"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MC",\n      "P1": "2900",\n      "P2": "3100",\n      "P3": "3000"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MA', 'P1': '3000', 'P2': '3200', 'P3': '3100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MB', 'P1': '2800', 'P2': '3300', 'P3': '2900'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MC', 'P1': '2900', 'P2': '3100', 'P3': '3000'}}]
from gurobipy import Model, GRB, quicksum

def build_and_solve_assignment():
    records = LEGACY_RECORDS
    managers = []
    projects = []
    cost = {}
    eligibility = {}
    for rec in records:
        vals = rec['values']
        manager = vals['Unnamed: 0']
        if manager not in managers:
            managers.append(manager)
        for proj in vals:
            if proj == 'Unnamed: 0':
                continue
            if proj not in projects:
                projects.append(proj)
    projects = [k for k in records[0]['values'] if k != 'Unnamed: 0']
    for rec in records:
        vals = rec['values']
        manager = vals['Unnamed: 0']
        for proj in projects:
            try:
                cij = float(vals[proj])
            except Exception:
                raise ValueError(f'Missing or invalid cost for ({manager},{proj})')
            cost[manager, proj] = cij
            eligibility[manager, proj] = 1
    if len(managers) != len(projects):
        raise ValueError('Number of managers and projects must be equal for assignment problem.')
    for i in managers:
        for j in projects:
            if (i, j) not in cost:
                raise ValueError(f'Missing cost for ({i},{j})')
            if (i, j) not in eligibility:
                raise ValueError(f'Missing eligibility for ({i},{j})')
    eligibility_mask = {(i, j): eligibility[i, j] for i in managers for j in projects}
    m = Model()
    x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, lb=0, ub=1, name='')
    for i in managers:
        for j in projects:
            if eligibility_mask[i, j] == 0:
                m.addConstr(x_vars[i, j] == 0, name='elig_%s_%s' % (i, j))
    for i in managers:
        m.addConstr(quicksum((x_vars[i, j] for j in projects)) == 1, name='mgr_%s' % i)
    for j in projects:
        m.addConstr(quicksum((x_vars[i, j] for i in managers)) == 1, name='prj_%s' % j)
    m.setObjective(quicksum((cost[i, j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for i in managers:
            for j in projects:
                v = x_vars[i, j]
                print(v.VarName, v.X)
    else:
        print('Solver status:', m.Status)
    return m
m = build_and_solve_assignment()