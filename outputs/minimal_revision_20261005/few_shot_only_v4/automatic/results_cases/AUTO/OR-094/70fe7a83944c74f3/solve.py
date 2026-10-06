import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Could not read CSV with any supported encoding.')
    if 'Workstation' not in df.columns:
        raise ValueError("Missing 'Workstation' column in CSV.")
    if 'Maintenance_Percent' not in df.columns:
        raise ValueError("Missing 'Maintenance_Percent' column in CSV.")
    workstations = [str(w) for w in df['Workstation'].unique()]
    radio_model_cols = [col for col in df.columns if col.startswith('HiFi') and col.endswith('_Minutes')]
    if len(radio_model_cols) != 101:
        raise ValueError(f'Expected 101 radio model columns, found {len(radio_model_cols)}.')
    radio_models = [col.replace('_Minutes', '') for col in radio_model_cols]
    t = {}
    for (idx, row) in df.iterrows():
        w = str(row['Workstation'])
        for col in radio_model_cols:
            m = col.replace('_Minutes', '')
            val = row[col]
            if pd.isnull(val):
                raise ValueError(f'Missing processing time for workstation {w}, model {m}.')
            t[w, m] = float(val)
    q = {}
    for (idx, row) in df.iterrows():
        w = str(row['Workstation'])
        val = row['Maintenance_Percent']
        if pd.isnull(val):
            raise ValueError(f'Missing Maintenance_Percent for workstation {w}.')
        q[w] = float(val)
    C = 1440.0
    E = {}
    for w in workstations:
        if w not in q:
            raise ValueError(f'Missing maintenance percent for workstation {w}.')
        E[w] = C * (1.0 - q[w] / 100.0)
    for w in workstations:
        for m in radio_models:
            if (w, m) not in t:
                raise ValueError(f'Missing processing time for workstation {w}, model {m}.')
    m = gp.Model('Radio_Idle_Min')
    x = m.addVars(radio_models, lb=0, vtype=GRB.INTEGER, name='')
    s = m.addVars(workstations, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((s[w] for w in workstations)), GRB.MINIMIZE)
    for w in workstations:
        m.addConstr(s[w] == E[w] - gp.quicksum((t[w, m] * x[m] for m in radio_models)), name=f'idle_def_{w}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()