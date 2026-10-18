import gurobipy as gp
import pandas as pd
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv'
df = pd.read_csv(path, sep=',')
df.set_index('Unnamed: 0', inplace=True)
managers = df.index.tolist()
projects = df.columns.tolist()
cost = {(m, p): df.loc[m, p] for m in managers for p in projects}
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY)
m.setObjective(gp.quicksum((cost[m, p] * x[m, p] for m in managers for p in projects)), gp.GRB.MINIMIZE)
for mng in managers:
    m.addConstr(x.sum(mng, '*') == 1)
for proj in projects:
    m.addConstr(x.sum('*', proj) == 1)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Assignments:')
    for mng in managers:
        for proj in projects:
            if x[mng, proj].X > 0.5:
                print(f'  Manager {mng} -> Project {proj}')
else:
    print(f'No optimal solution found. Status: {m.status}')