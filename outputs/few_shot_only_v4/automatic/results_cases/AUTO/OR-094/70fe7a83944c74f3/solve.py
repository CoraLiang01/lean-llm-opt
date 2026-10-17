import gurobipy as gp
from gurobipy import GRB
import pandas as pd
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
model_cols = [col for col in df.columns if col.endswith('_Minutes')]
models = [col.replace('_Minutes', '') for col in model_cols]
if len(models) != 101:
    raise ValueError(f'Expected 101 models, found {len(models)}')
workstations = ['1', '2', '3']
t = {}
for w_idx, w in enumerate(workstations):
    row = df.iloc[w_idx]
    t[w] = {}
    for m_idx, model in enumerate(models):
        val = row[model_cols[m_idx]]
        try:
            t[w][model] = float(val)
        except Exception:
            raise ValueError(f'Invalid processing time for workstation {w}, model {model}: {val}')
C = {'1': 1440 * (1 - 0.1), '2': 1440 * (1 - 0.14), '3': 1440 * (1 - 0.12)}
m = gp.Model('Radio_Idle_Min')
x_vars = m.addVars(models, lb=0, vtype=GRB.INTEGER, name='')
I_vars = m.addVars(workstations, lb=0, vtype=GRB.CONTINUOUS, name='')
for w in workstations:
    m.addConstr(I_vars[w] == C[w] - gp.quicksum((t[w][model] * x_vars[model] for model in models)), name=f'idle_def_{w}')
    m.addConstr(gp.quicksum((t[w][model] * x_vars[model] for model in models)) <= C[w], name=f'cap_{w}')
m.setObjective(gp.quicksum((I_vars[w] for w in workstations)), GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')