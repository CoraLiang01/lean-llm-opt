import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
machine_ids = cost_df['Machine'].tolist()
task_ids = [col for col in cost_df.columns if col != 'Machine']
if len(machine_ids) != 12 or len(task_ids) != 12:
    raise ValueError(f'Expected 12 machines and 12 tasks, got {len(machine_ids)} machines and {len(task_ids)} tasks.')
cost = {}
for (idx, row) in cost_df.iterrows():
    machine = str(row['Machine'])
    for task in task_ids:
        try:
            cij = float(row[task])
        except Exception as e:
            raise ValueError(f"Invalid cost for machine '{machine}', task '{task}': {row[task]}") from e
        cost[machine, task] = cij
for machine in machine_ids:
    for task in task_ids:
        if (machine, task) not in cost:
            raise ValueError(f"Missing cost for machine '{machine}', task '{task}'.")
m = gp.Model('AssignmentProblem')
x_vars = m.addVars([(machine, task) for machine in machine_ids for task in task_ids], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[machine, task] * x_vars[machine, task] for machine in machine_ids for task in task_ids)), gp.GRB.MINIMIZE)
for machine in machine_ids:
    m.addConstr(gp.quicksum((x_vars[machine, task] for task in task_ids)) == 1, name=f'mach_{machine}')
for task in task_ids:
    m.addConstr(gp.quicksum((x_vars[machine, task] for machine in machine_ids)) == 1, name=f'task_{task}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for (machine, task) in x_vars.keys():
        var = x_vars[machine, task]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')