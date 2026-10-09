import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activities_df['Activity'] = activities_df['Activity'].str.strip()
activities = list(activities_df['Activity'])
normal_duration = {}
crash_duration = {}
crash_cost_per_day = {}
crash_day_upper = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    nd = int(row['NormalDuration'])
    cd = int(row['CrashDuration'])
    crash_cost = int(row['CrashCostPerDay'])
    normal_duration[act] = nd
    crash_duration[act] = cd
    crash_cost_per_day[act] = crash_cost
    crash_day_upper[act] = nd - cd
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    preds_raw = row['Predecessors'].strip()
    if preds_raw == '':
        predecessors[act] = []
    else:
        preds = [p.strip() for p in preds_raw.split(';') if p.strip() != '']
        predecessors[act] = preds
deadline_row = parameters_df[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activities, lb=0, ub=[crash_day_upper[act] for act in activities], vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[act] * z_vars[act] for act in activities)), gp.GRB.MINIMIZE)
for i in activities:
    for j in predecessors[i]:
        if j not in activities:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activities:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'completion_{i}')
m.addConstr(T_var <= project_deadline, name='deadline')
m.optimize()