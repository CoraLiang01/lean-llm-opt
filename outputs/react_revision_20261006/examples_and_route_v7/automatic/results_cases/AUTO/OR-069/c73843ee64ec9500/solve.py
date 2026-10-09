import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_assignment_problem(costs_csv_path):
    df = pd.read_csv(costs_csv_path, dtype=str, keep_default_na=False)
    managers = df['Manager'].tolist()
    if len(set(managers)) != len(managers):
        raise ValueError('Duplicate manager identifiers found in the CSV.')
    project_cost_col_pattern = re.compile('^Project\\s+(\\d+)\\s+Cost$', re.IGNORECASE)
    project_cols = []
    project_ids = []
    for col in df.columns:
        m = project_cost_col_pattern.match(col)
        if m:
            project_cols.append(col)
            project_ids.append(m.group(1))
    if len(project_cols) == 0:
        raise ValueError('No project cost columns found in the CSV.')
    if len(set(project_ids)) != len(project_ids):
        raise ValueError('Duplicate project identifiers found in the CSV.')
    if len(managers) != len(project_ids):
        raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(project_ids)}).')
    cost = {}
    for (idx, row) in df.iterrows():
        manager = row['Manager']
        for (col, project) in zip(project_cols, project_ids):
            val = row[col]
            if val == '':
                raise ValueError(f"Missing cost for manager '{manager}', project '{project}'.")
            try:
                cij = int(val)
            except Exception:
                raise ValueError(f"Non-integer cost '{val}' for manager '{manager}', project '{project}'.")
            cost[manager, project] = cij
    for manager in managers:
        for project in project_ids:
            if (manager, project) not in cost:
                raise ValueError(f"Missing cost entry for manager '{manager}', project '{project}'.")
    m = gp.Model('ManagerProjectAssignment')
    m.Params.MIPGap = 0.0001
    x_keys = [(manager, project) for manager in managers for project in project_ids]
    x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[manager, project] * x_vars[manager, project] for manager in managers for project in project_ids)), gp.GRB.MINIMIZE)
    for manager in managers:
        m.addConstr(gp.quicksum((x_vars[manager, project] for project in project_ids)) == 1, name=f'mgr_{manager}')
    for project in project_ids:
        m.addConstr(gp.quicksum((x_vars[manager, project] for manager in managers)) == 1, name=f'prj_{project}')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for (manager, project) in x_keys:
            var = x_vars[manager, project]
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_assignment_problem('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv')