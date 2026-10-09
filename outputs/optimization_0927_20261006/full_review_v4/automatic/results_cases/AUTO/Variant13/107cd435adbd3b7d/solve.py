import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].tolist()

def to_int_series(df, col):
    return df.set_index('Activity')[col].astype(int).to_dict()
NormalDuration = to_int_series(activities_df, 'NormalDuration')
CrashDuration = to_int_series(activities_df, 'CrashDuration')
CrashCostPerDay = to_int_series(activities_df, 'CrashCostPerDay')
CrashMax = {i: NormalDuration[i] - CrashDuration[i] for i in activity_ids}

def parse_preds(s):
    s = s.strip()
    if not s:
        return []
    return [pred.strip() for pred in s.split(';') if pred.strip()]
Predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    preds = parse_preds(row['Predecessors'])
    Predecessors[act] = preds
deadline_row = parameters_df[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, ub={i: CrashMax[i] for i in activity_ids}, vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((CrashCostPerDay[i] * z_vars[i] for i in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in Predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (NormalDuration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_ids:
    m.addConstr(s_vars[i] + (NormalDuration[i] - z_vars[i]) <= T_var, name=f'completion_{i}')
m.addConstr(T_var <= ProjectDeadline, name='deadline')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total crash cost: {m.objVal:.2f}')
    print(f'Project completion time (T): {T_var.X:.2f}')
    print('\n--- Activity Schedule ---')
    for i in activity_ids:
        print(f'Activity {i}:')
        print(f'  Start time (s_{i}): {s_vars[i].X:.2f}')
        print(f'  Crash days used (z_{i}): {int(round(z_vars[i].X))} (max {CrashMax[i]})')
        print(f'  Duration: {NormalDuration[i] - int(round(z_vars[i].X))} days (Normal: {NormalDuration[i]}, Crash: {CrashDuration[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')