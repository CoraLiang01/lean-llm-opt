import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.fullmatch(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")

def solve_project_selection():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
    df = pd.read_csv(path, sep=',', dtype=str, keep_default_na=False)
    id_col = 'Project ID'
    name_col = 'Project Name'
    capital_col = 'Capital (k$)'
    npv_col = 'NPV (k$)'
    df[id_col] = df[id_col].astype(str).str.strip()
    df[name_col] = df[name_col].astype(str).str.strip()
    df[capital_col] = df[capital_col].astype(str).str.strip()
    df[npv_col] = df[npv_col].astype(str).str.strip()
    project_ids = [str(i) for i in range(1, 111)]
    df = df[df[id_col].isin(project_ids)].copy()
    if len(df) != 110:
        raise ValueError(f'Expected 110 projects with IDs 1..110, found {len(df)}.')
    projects = [int(pid) for pid in df[id_col]]
    projects_set = set(projects)
    if set(range(1, 111)) != projects_set:
        raise ValueError('Project IDs in CSV do not exactly match 1..110.')
    npv = {}
    capital = {}
    name = {}
    for (_, row) in df.iterrows():
        pid = int(row[id_col])
        try:
            npv[pid] = float(row[npv_col])
            capital[pid] = float(row[capital_col])
            name[pid] = row[name_col]
        except Exception as e:
            raise ValueError(f'Invalid numeric data for project {pid}: {e}')
    m = gp.Model('ProjectSelection')
    x_vars = m.addVars(projects, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in projects)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in projects)) <= 1000, name='budget')
    if 4 in projects and 7 in projects:
        m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_4_7')
    else:
        raise ValueError('Projects 4 and 7 must be present for mutual exclusion constraint.')
    if 6 in projects and 1 in projects:
        m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_1')
    else:
        raise ValueError('Projects 6 and 1 must be present for pre-requisite constraint.')
    if 10 in projects and 5 in projects:
        m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_5')
    else:
        raise ValueError('Projects 10 and 5 must be present for contingent constraint.')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in projects:
            print(f'x[{i}] {x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_project_selection()