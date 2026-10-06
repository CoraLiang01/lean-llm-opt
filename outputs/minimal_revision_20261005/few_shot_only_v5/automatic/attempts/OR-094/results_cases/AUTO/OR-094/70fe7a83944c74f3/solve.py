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
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with any of the specified encodings.')
    if 'Workstation' not in df.columns or 'Maintenance_Percent' not in df.columns:
        raise ValueError('Missing required columns in CSV.')
    workstations = df['Workstation'].tolist()
    hifi_cols = [col for col in df.columns if col.startswith('HiFi') and col.endswith('_Minutes')]
    if len(hifi_cols) != 101:
        raise ValueError('Expected 101 HiFi columns, found {}'.format(len(hifi_cols)))
    models = [col.replace('_Minutes', '') for col in hifi_cols]
    t = {}
    for (idx, row) in df.iterrows():
        w = row['Workstation']
        t[w] = {}
        for col in hifi_cols:
            t[w][col.replace('_Minutes', '')] = row[col]
    total_minutes = 1440
    c_eff = {}
    for (idx, row) in df.iterrows():
        w = row['Workstation']
        maint = row['Maintenance_Percent']
        if pd.isnull(maint):
            raise ValueError(f'Missing Maintenance_Percent for workstation {w}')
        c_eff[w] = total_minutes * (1 - float(maint) / 100)
    m = gp.Model('Minimize_Idle_Time')
    x = m.addVars(models, lb=0, vtype=GRB.INTEGER, name='')
    I = m.addVars(workstations, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((I[w] for w in workstations)), GRB.MINIMIZE)
    for w in workstations:
        m.addConstr(I[w] >= c_eff[w] - gp.quicksum((t[w][m] * x[m] for m in models)), name=f'idle_def_{w}')
    for w in workstations:
        m.addConstr(gp.quicksum((t[w][m] * x[m] for m in models)) <= c_eff[w], name=f'capacity_{w}')
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