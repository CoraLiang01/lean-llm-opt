import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    managers = df['Manager'].tolist()
    projects = [c for c in df.columns if c != 'Manager']
    cost = {(row['Manager'], proj): float(row[proj]) for row in df.to_dict('records') for proj in projects}
    for mgr in managers:
        for proj in projects:
            if (mgr, proj) not in cost:
                raise ValueError(f'Missing cost for manager {mgr} and project {proj}')
    model = gurobipy.Model('ManagerProjectAssignment')
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(mgr, proj) for mgr in managers for proj in projects], vtype=GRB.BINARY, name='')
    model.setObjective(gurobipy.quicksum((cost[mgr, proj] * x[mgr, proj] for mgr in managers for proj in projects)), GRB.MINIMIZE)
    for mgr in managers:
        model.addConstr(gurobipy.quicksum((x[mgr, proj] for proj in projects)) == 1, name=f'manager_{mgr}')
    for proj in projects:
        model.addConstr(gurobipy.quicksum((x[mgr, proj] for mgr in managers)) == 1, name=f'project_{proj}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(model.ObjVal)
        for var in model.getVars():
            print(var.VarName, var.X)
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()