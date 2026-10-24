import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_path = 'benchmark_dataset/Variant13/inputs/project_activities.csv'
parameters_path = 'benchmark_dataset/Variant13/inputs/project_parameters.csv'
activities_df = pd.read_csv(activities_path, sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv(parameters_path, sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].tolist()
for col in ['NormalDuration', 'CrashDuration', 'CrashCostPerDay']:
    activities_df[col] = activities_df[col].astype(int)
crash_lb = {row['Activity']: 0 for (_, row) in activities_df.iterrows()}
crash_ub = {row['Activity']: row['NormalDuration'] - row['CrashDuration'] for (_, row) in activities_df.iterrows()}
crash_cost = {row['Activity']: row['CrashCostPerDay'] for (_, row) in activities_df.iterrows()}
normal_duration = {row['Activity']: row['NormalDuration'] for (_, row) in activities_df.iterrows()}
crash_duration = {row['Activity']: row['CrashDuration'] for (_, row) in activities_df.iterrows()}

def parse_preds(cell):
    cell = cell.strip()
    if cell == '':
        return []
    return [pred.strip() for pred in cell.split(';') if pred.strip() != '']
predecessors = {row['Activity']: parse_preds(row['Predecessors']) for (_, row) in activities_df.iterrows()}
deadline_row = parameters_df[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, ub=[crash_ub[act] for act in activity_ids], vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost[act] * z_vars[act] for act in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_ids:
    m.addConstr(z_vars[i] >= crash_lb[i], name=f'crash_lb_{i}')
    m.addConstr(z_vars[i] <= crash_ub[i], name=f'crash_ub_{i}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'T_completion_{i}')
m.addConstr(T_var <= project_deadline, name='project_deadline')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total crashing cost: {m.objVal:.2f}')
    print(f'Project completion time (T): {T_var.X:.2f}')
    print('\n--- Activity Schedule ---')
    for i in activity_ids:
        print(f'Activity {i}:')
        print(f'  Start time (s_{i}): {s_vars[i].X:.2f}')
        print(f'  Crash days used (z_{i}): {int(round(z_vars[i].X))} (max {crash_ub[i]})')
        print(f'  Duration: {normal_duration[i] - int(round(z_vars[i].X))} days (Normal: {normal_duration[i]}, Crash: {crash_duration[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')