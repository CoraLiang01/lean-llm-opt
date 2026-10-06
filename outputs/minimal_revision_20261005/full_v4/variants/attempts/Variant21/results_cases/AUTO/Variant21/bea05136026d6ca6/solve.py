import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_project_crashing():
    activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',')
    params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',')
    activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
    activities = list(activities_df['Activity'])
    normal_duration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
    crash_duration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
    crash_cost_per_day = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
    crash_day_ub = {i: int(normal_duration[i] - crash_duration[i]) for i in activities}
    crash_day_lb = {i: 0 for i in activities}
    pred_dict = {}
    for (idx, row) in activities_df.iterrows():
        preds = []
        if isinstance(row['Predecessors'], str) and row['Predecessors'].strip() != '':
            preds = [p.strip() for p in row['Predecessors'].split(';') if p.strip() != '']
        pred_dict[row['Activity']] = preds
    deadline_row = params_df[params_df['Parameter'].str.casefold() == 'projectdeadline']
    if deadline_row.empty:
        raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
    project_deadline = int(deadline_row['Value'].iloc[0])
    for i in activities:
        if i not in normal_duration or i not in crash_duration or i not in crash_cost_per_day:
            raise ValueError(f'Missing duration or cost data for activity {i}')
        if crash_day_ub[i] < 0:
            raise ValueError(f'Crash duration for activity {i} exceeds normal duration')
    m = gp.Model('ProjectCrashing')
    m.setParam('MIPGap', 0.0001)
    s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z = m.addVars(activities, lb=[crash_day_lb[i] for i in activities], ub=[crash_day_ub[i] for i in activities], vtype=gp.GRB.INTEGER, name='')
    T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((crash_cost_per_day[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
    for i in activities:
        for j in pred_dict[i]:
            if j not in activities:
                raise ValueError(f'Predecessor {j} of activity {i} not found in activity list')
            m.addConstr(s[i] >= s[j] + (normal_duration[j] - z[j]), name=f'prec_{j}_to_{i}')
    for i in activities:
        m.addConstr(T >= s[i] + (normal_duration[i] - z[i]), name=f'comp_{i}')
    m.addConstr(T <= project_deadline, name='deadline')
    m.optimize()
    return m
m = solve_project_crashing()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')