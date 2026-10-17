import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Owner' not in df.columns:
    raise KeyError("CSV must have 'Owner' as the first column.")
worker_list = [col for col in df.columns if col != 'Owner']
owner_list = df['Owner'].tolist()
if set(owner_list) != set(worker_list):
    raise ValueError('Mismatch between owners and workers. Each owner must correspond to a worker.')
work_days = {}
for i, owner in enumerate(owner_list):
    work_days[owner] = {}
    for worker in worker_list:
        val = df.iloc[i][worker].strip()
        if val == '':
            days = 0
        else:
            try:
                days = int(val)
            except Exception as e:
                raise ValueError(f"Non-integer value in work_days at owner '{owner}', worker '{worker}': '{val}'")
        work_days[owner][worker] = days
for worker in worker_list:
    total_days = sum((work_days[owner][worker] for owner in owner_list))
    if total_days != 10:
        raise ValueError(f"Worker '{worker}' has total contributed days {total_days}, expected 10.")
m = gp.Model('Mutual_Payment_Wages')
w_vars = m.addVars(worker_list, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for k in worker_list:
    total_paid = gp.quicksum((work_days[k][j] * w_vars[j] for j in worker_list))
    total_earned = gp.quicksum((work_days[i][k] for i in owner_list if i != k)) * w_vars[k]
    m.addConstr(total_paid == total_earned, name=f'balance_{k}')
first_worker = worker_list[0]
m.addConstr(w_vars[first_worker] == 60.0, name='fixed_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found:')
    for worker in worker_list:
        print(f'{worker}: {w_vars[worker].X:.6f}')
else:
    print(f'No feasible solution found. Status: {m.status}')