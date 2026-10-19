import gurobipy as gp
import pandas as pd
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv'
df = pd.read_csv(csv_path)
managers = df['Manager'].tolist()
project_cols = [col for col in df.columns if col != 'Manager']
project_names = [col.replace(' Cost', '') for col in project_cols]
cost = {}
for mgr in managers:
    cost[mgr] = {}
    for col, proj in zip(project_cols, project_names):
        cost[mgr][proj] = df.loc[df['Manager'] == mgr, col].iloc[0]
m = gp.Model('Assignment')
x = m.addVars(managers, project_names, vtype=gp.GRB.BINARY)
for mgr in managers:
    m.addConstr(x.sum(mgr, '*') == 1)
for proj in project_names:
    m.addConstr(x.sum('*', proj) == 1)
m.setObjective(gp.quicksum((cost[mgr][proj] * x[mgr, proj] for mgr in managers for proj in project_names)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Assignments:')
    for mgr in managers:
        for proj in project_names:
            if x[mgr, proj].X > 0.5:
                print(f'  {mgr} -> {proj} (cost {cost[mgr][proj]})')
else:
    print(f'No optimal solution found. Status: {m.status}')