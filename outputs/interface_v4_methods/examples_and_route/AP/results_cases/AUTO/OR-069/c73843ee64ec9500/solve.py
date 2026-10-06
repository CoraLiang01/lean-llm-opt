import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_assignment_problem(costs_csv_path):
    df = pd.read_csv(costs_csv_path, sep=',')
    managers = df['Manager'].astype(str).tolist()
    project_cost_col_pattern = re.compile('^Project\\s+(\\d+)\\s+Cost$', re.IGNORECASE)
    project_cols = []
    project_ids = []
    for col in df.columns:
        m = project_cost_col_pattern.match(col.strip())
        if m:
            project_cols.append(col)
            project_ids.append(f'Project {m.group(1)}')
    if len(managers) != len(project_ids):
        raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(project_ids)}).')
    col_to_proj = dict(zip(project_cols, project_ids))
    cost = {}
    for idx, row in df.iterrows():
        manager = str(row['Manager'])
        cost[manager] = {}
        for col in project_cols:
            project = col_to_proj[col]
            val = row[col]
            if pd.isnull(val):
                raise ValueError(f"Missing cost for manager '{manager}', project '{project}'.")
            cost[manager][project] = float(val)
    m = gp.Model('ManagerProjectAssignment')
    x = m.addVars(managers, project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in managers for j in project_ids)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for j in project_ids)) == 1 for i in managers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for i in managers)) == 1 for j in project_ids), name='')
    m.optimize()
    return m
m = solve_assignment_problem('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv')