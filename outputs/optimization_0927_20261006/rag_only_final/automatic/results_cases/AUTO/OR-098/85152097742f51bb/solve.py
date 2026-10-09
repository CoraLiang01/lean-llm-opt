import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
worker_cols = [col for col in df.columns if col != 'Owner']
workers = list(worker_cols)
owners = df['Owner'].tolist()
if set(owners) != set(workers):
    raise ValueError('Mismatch between owners and workers. Each owner must be a worker and vice versa.')
days = {}
for row in df.itertuples(index=False):
    owner = getattr(row, 'Owner')
    for (j, worker) in enumerate(worker_cols):
        val = getattr(row, worker)
        try:
            days[owner, worker] = int(val)
        except Exception as e:
            raise ValueError(f'Invalid value for days at owner={owner}, worker={worker}: {val}') from e
m = Model('mutual_wage_balance')
m.Params.OutputFlag = 0
wage_vars = m.addVars(workers, vtype=GRB.CONTINUOUS, name='')
first_worker = workers[0]
if first_worker != 'Carpenter':
    raise ValueError(f"First worker in file is '{first_worker}', expected 'Carpenter'.")
m.addConstr(wage_vars[first_worker] == 60.0, name='fix_carpenter_wage')
for i in workers:
    income = quicksum((days[o, i] * wage_vars[i] for o in owners if o != i))
    expense = quicksum((days[i, j] * wage_vars[j] for j in workers if j != i))
    m.addConstr(income == expense, name=f'balance_{i}')
m.setObjective(0, GRB.MINIMIZE)
m.optimize()