import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
all_columns = list(df.columns)
if 'Owner' not in all_columns:
    raise ValueError("CSV missing required 'Owner' column")
workers = [col for col in all_columns if col != 'Owner']
homeowners = df['Owner'].tolist()
if len(workers) != len(homeowners):
    raise ValueError('Number of workers (columns) does not match number of homeowners (rows)')
for w in workers:
    df[w] = df[w].astype(int)
days = {}
for i, row in df.iterrows():
    owner = row['Owner']
    days[owner] = {}
    for w in workers:
        days[owner][w] = int(row[w])
for w in workers:
    total_days = sum((days[owner][w] for owner in homeowners))
    if total_days != 10:
        raise ValueError(f'Worker {w} has total work days {total_days}, expected 10.')
first_worker = workers[0]
fixed_wage = 60.0
m = Model('mutual_wage_balance')
m.Params.OutputFlag = 0
wage_vars = m.addVars(workers, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='')
wage_vars[first_worker].lb = fixed_wage
wage_vars[first_worker].ub = fixed_wage
for k in workers:
    lhs = quicksum((days[owner][k] * wage_vars[k] for owner in homeowners))
    rhs = quicksum((days[k][j] * wage_vars[j] for j in workers))
    m.addConstr(lhs == rhs, name='fairness_' + k)
m.setObjective(0, GRB.MINIMIZE)
m.optimize()