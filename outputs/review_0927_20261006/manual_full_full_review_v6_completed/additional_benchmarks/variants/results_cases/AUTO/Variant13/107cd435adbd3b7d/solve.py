import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_predecessors(s):
    if pd.isna(s) or str(s).strip() == '':
        return []
    return [x.strip() for x in str(s).split(';') if x.strip() != '']
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].tolist()

def to_int_series(df, col):
    return df[col].astype(int)
normal_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'NormalDuration')))
crash_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashDuration')))
crash_cost_per_day = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashCostPerDay')))
predecessors = {}
for idx, row in activities_df.iterrows():
    act = row['Activity']
    preds = parse_predecessors(row['Predecessors'])
    predecessors[act] = preds
deadline_row = params_df.loc[params_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = {}
for i in activity_ids:
    lb = 0
    ub = normal_duration[i] - crash_duration[i]
    z_vars[i] = m.addVar(lb=lb, ub=ub, vtype=gp.GRB.INTEGER, name=f'z_{i}')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.update()
m.setObjective(gp.quicksum((crash_cost_per_day[i] * z_vars[i] for i in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'completion_{i}')
m.addConstr(T_var <= project_deadline, name='deadline')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total crashing cost: {m.objVal:.2f}')
    print(f'Project completion time (T): {T_var.X:.2f}')
    print('\n--- Activity Schedule ---')
    for i in activity_ids:
        print(f'Activity {i}:')
        print(f'  Start time (s_{i}): {s_vars[i].X:.2f}')
        print(f'  Crash days used (z_{i}): {int(round(z_vars[i].X))} (max {normal_duration[i] - crash_duration[i]})')
        print(f'  Duration: {normal_duration[i] - int(round(z_vars[i].X))} days')
else:
    print(f'No optimal solution found. Status: {m.status}')