import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {csv_path} with tried encodings.')
    workstation_col = 'Workstation'
    maintenance_col = 'Maintenance_Percent'
    model_cols = [col for col in df.columns if col.startswith('HiFi') and col.endswith('_Minutes')]
    if not model_cols:
        raise ValueError('No model columns found in CSV.')
    models = [col.replace('_Minutes', '') for col in model_cols]
    workstations = df[workstation_col].tolist()
    if len(workstations) != 3:
        raise ValueError('Expected 3 workstations, found: %s' % workstations)
    t = {}
    for (idx, row) in df.iterrows():
        w = row[workstation_col]
        for (m_col, m) in zip(model_cols, models):
            val = row[m_col]
            try:
                t[w, m] = float(val)
            except Exception:
                raise ValueError(f'Invalid processing time for workstation {w}, model {m}: {val}')
    total_minutes = 1440
    E = {}
    for (idx, row) in df.iterrows():
        w = row[workstation_col]
        maint = row[maintenance_col]
        try:
            maint_pct = float(maint)
        except Exception:
            raise ValueError(f'Invalid maintenance percent for workstation {w}: {maint}')
        E[w] = total_minutes * (1 - maint_pct / 100.0)
    for w in workstations:
        for m in models:
            if (w, m) not in t:
                raise ValueError(f'Missing processing time for workstation {w}, model {m}')
        if w not in E:
            raise ValueError(f'Missing effective capacity for workstation {w}')
    m = gp.Model('radio_idle_time_min')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(models, lb=0, vtype=GRB.INTEGER, name='')
    s_vars = m.addVars(workstations, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((s_vars[w] for w in workstations)), GRB.MINIMIZE)
    for w in workstations:
        m.addConstr(gp.quicksum((t[w, m] * x_vars[m] for m in models)) + s_vars[w] == E[w], name=f'time_balance_{w}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')