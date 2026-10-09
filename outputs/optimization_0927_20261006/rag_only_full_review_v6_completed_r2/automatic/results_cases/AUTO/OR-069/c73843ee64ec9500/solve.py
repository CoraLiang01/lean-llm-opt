from gurobipy import Model, GRB, quicksum
managers = ['Manager 1', 'Manager 2', 'Manager 3', 'Manager 4', 'Manager 5', 'Manager 6', 'Manager 7', 'Manager 8', 'Manager 9', 'Manager 10', 'Manager 11']
projects = ['Project 1', 'Project 2', 'Project 3', 'Project 4', 'Project 5', 'Project 6', 'Project 7', 'Project 8', 'Project 9', 'Project 10', 'Project 11']
c_ij = {'Manager 1': {'Project 1': 12, 'Project 2': 7, 'Project 3': 9, 'Project 4': 7, 'Project 5': 9, 'Project 6': 8, 'Project 7': 7, 'Project 8': 6, 'Project 9': 8, 'Project 10': 7, 'Project 11': 8}, 'Manager 2': {'Project 1': 8, 'Project 2': 9, 'Project 3': 6, 'Project 4': 6, 'Project 5': 6, 'Project 6': 7, 'Project 7': 8, 'Project 8': 7, 'Project 9': 7, 'Project 10': 8, 'Project 11': 9}, 'Manager 3': {'Project 1': 7, 'Project 2': 17, 'Project 3': 12, 'Project 4': 14, 'Project 5': 10, 'Project 6': 9, 'Project 7': 8, 'Project 8': 8, 'Project 9': 7, 'Project 10': 8, 'Project 11': 9}, 'Manager 4': {'Project 1': 15, 'Project 2': 14, 'Project 3': 6, 'Project 4': 6, 'Project 5': 8, 'Project 6': 7, 'Project 7': 8, 'Project 8': 9, 'Project 9': 8, 'Project 10': 7, 'Project 11': 8}, 'Manager 5': {'Project 1': 8, 'Project 2': 7, 'Project 3': 8, 'Project 4': 9, 'Project 5': 6, 'Project 6': 8, 'Project 7': 7, 'Project 8': 8, 'Project 9': 7, 'Project 10': 8, 'Project 11': 7}, 'Manager 6': {'Project 1': 7, 'Project 2': 8, 'Project 3': 7, 'Project 4': 8, 'Project 5': 7, 'Project 6': 9, 'Project 7': 8, 'Project 8': 7, 'Project 9': 8, 'Project 10': 7, 'Project 11': 8}, 'Manager 7': {'Project 1': 9, 'Project 2': 6, 'Project 3': 8, 'Project 4': 7, 'Project 5': 8, 'Project 6': 7, 'Project 7': 9, 'Project 8': 8, 'Project 9': 7, 'Project 10': 8, 'Project 11': 7}, 'Manager 8': {'Project 1': 8, 'Project 2': 7, 'Project 3': 7, 'Project 4': 8, 'Project 5': 7, 'Project 6': 8, 'Project 7': 7, 'Project 8': 9, 'Project 9': 8, 'Project 10': 7, 'Project 11': 8}, 'Manager 9': {'Project 1': 7, 'Project 2': 8, 'Project 3': 8, 'Project 4': 7, 'Project 5': 8, 'Project 6': 7, 'Project 7': 8, 'Project 8': 7, 'Project 9': 9, 'Project 10': 8, 'Project 11': 7}, 'Manager 10': {'Project 1': 8, 'Project 2': 7, 'Project 3': 8, 'Project 4': 7, 'Project 5': 8, 'Project 6': 7, 'Project 7': 8, 'Project 8': 7, 'Project 9': 8, 'Project 10': 9, 'Project 11': 8}, 'Manager 11': {'Project 1': 7, 'Project 2': 8, 'Project 3': 7, 'Project 4': 8, 'Project 5': 7, 'Project 6': 8, 'Project 7': 7, 'Project 8': 8, 'Project 9': 7, 'Project 10': 8, 'Project 11': 9}}
if set(c_ij.keys()) != set(managers):
    raise ValueError('Cost matrix manager keys do not match manager set.')
for m in managers:
    if set(c_ij[m].keys()) != set(projects):
        raise ValueError(f'Cost matrix project keys for {m} do not match project set.')

def build_assignment_model():
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x_ij_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
    m.addConstrs((quicksum((x_ij_vars[manager, project] for project in projects)) == 1 for manager in managers), name='')
    m.addConstrs((quicksum((x_ij_vars[manager, project] for manager in managers)) == 1 for project in projects), name='')
    m.setObjective(quicksum((c_ij[manager][project] * x_ij_vars[manager, project] for manager in managers for project in projects)), GRB.MINIMIZE)
    return m
m = build_assignment_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status', m.Status)