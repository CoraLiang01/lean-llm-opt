import gurobipy as gp
from gurobipy import GRB
import pandas as pd
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
model_cols = [col for col in df.columns if col.startswith('HiFi') and col.endswith('_Minutes')]
if len(model_cols) != 101:
    raise ValueError(f'Expected 101 model columns, found {len(model_cols)}')
models = [col.replace('_Minutes', '') for col in model_cols]
workstation_rows = df['Workstation'].tolist()
if workstation_rows != ['1', '2', '3']:
    raise ValueError(f'Expected Workstation rows 1,2,3, got {workstation_rows}')
workstations = ['1', '2', '3']
a = {}
for w_idx, w in enumerate(workstations):
    a[w] = {}
    for m_idx, model_col in enumerate(model_cols):
        try:
            a[w][models[m_idx]] = int(df.loc[w_idx, model_col])
        except Exception as e:
            raise ValueError(f'Error parsing a[{w}][{models[m_idx]}]: {e}')
C = {'1': 1440 * (1 - 0.1), '2': 1440 * (1 - 0.14), '3': 1440 * (1 - 0.12)}
C = {w: float(f'{C[w]:.4f}') for w in workstations}
m = gp.Model('radio_idle_time_min')
x_vars = m.addVars(models, lb=0, vtype=GRB.INTEGER, name='')
I_vars = m.addVars(workstations, lb=0.0, vtype=GRB.CONTINUOUS, name='')
for w in workstations:
    m.addConstr(gp.quicksum((a[w][k] * x_vars[k] for k in models)) + I_vars[w] == C[w], name=f'capacity_{w}')
m.setObjective(gp.quicksum((I_vars[w] for w in workstations)), GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')