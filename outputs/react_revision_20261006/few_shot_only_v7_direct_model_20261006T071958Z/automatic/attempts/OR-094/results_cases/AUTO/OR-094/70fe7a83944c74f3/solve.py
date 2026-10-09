import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Could not read CSV with any supported encoding.')
    model_cols = [col for col in df.columns if col.casefold().startswith('hifi') and col.casefold().endswith('_minutes')]
    if len(model_cols) != 101:
        raise ValueError(f'Expected 101 HiFi*_Minutes columns, found {len(model_cols)}.')
    M = model_cols.copy()
    if 'Workstation' not in df.columns:
        raise ValueError("Missing 'Workstation' column in input data.")
    W = list(df['Workstation'].unique())
    W.sort(key=lambda x: int(x) if x.isdigit() else x)
    t = {}
    for (_, row) in df.iterrows():
        w = row['Workstation']
        for m in M:
            try:
                val = float(row[m])
            except Exception:
                raise ValueError(f'Non-numeric or missing processing time for workstation {w}, model {m}.')
            t[w, m] = val
    C = {w: 1440.0 for w in W}
    if 'Maintenance_Percent' not in df.columns:
        raise ValueError("Missing 'Maintenance_Percent' column in input data.")
    r = {}
    for (_, row) in df.iterrows():
        w = row['Workstation']
        try:
            r_w = float(row['Maintenance_Percent'])
        except Exception:
            raise ValueError(f'Non-numeric or missing Maintenance_Percent for workstation {w}.')
        r[w] = r_w
    E = {w: C[w] * (1 - r[w] / 100.0) for w in W}
    m = gp.Model('Minimize_Total_Idle_Time')
    quantity_vars = m.addVars(M, lb=0, vtype=GRB.INTEGER, name='')
    idle_vars = m.addVars(W, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((idle_vars[w] for w in W)), GRB.MINIMIZE)
    for w in W:
        m.addConstr(idle_vars[w] == E[w] - gp.quicksum((t[w, m_key] * quantity_vars[m_key] for m_key in M)), name=f'idle_def_{w}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')