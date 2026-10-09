import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_project_id_by_name(df, target_name):

    def normalize(s):
        return re.sub('\\s+', ' ', str(s)).strip().casefold()
    norm_target = normalize(target_name)
    matches = df[df['Project Name'].apply(lambda x: normalize(x) == norm_target)]
    if len(matches) == 0:
        raise ValueError(f"Project with name '{target_name}' not found in project.csv")
    if len(matches) > 1:
        raise ValueError(f"Multiple projects found for name '{target_name}'")
    return int(matches.iloc[0]['Project ID'])

def solve_project_selection():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',', dtype=str, keep_default_na=False)
    project_ids = df['Project ID'].astype(int).tolist()
    n_projects = len(project_ids)
    if n_projects != 110:
        raise ValueError(f'Expected 110 projects, found {n_projects}')
    npv_dict = {}
    capital_dict = {}
    for (idx, row) in df.iterrows():
        pid = int(row['Project ID'])
        try:
            npv = int(row['NPV (k$)'])
            capital = int(row['Capital (k$)'])
        except Exception as e:
            raise ValueError(f'Non-numeric NPV or Capital for Project ID {pid}: {e}')
        npv_dict[pid] = npv
        capital_dict[pid] = capital
    id_rnd_alpha = find_project_id_by_name(df, 'R&D Initiative Alpha')
    id_global_expansion = find_project_id_by_name(df, 'Global Expansion Pilot')
    id_infra_upgrade = find_project_id_by_name(df, 'Infrastructure Upgrade')
    id_sys_auto = find_project_id_by_name(df, 'System Automation')
    id_staff_training = find_project_id_by_name(df, 'Staff Training Program')
    id_cust_exp_platform = find_project_id_by_name(df, 'Customer Experience Platform')
    m = gp.Model('ProjectSelection')
    x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((x_vars[pid] * npv_dict[pid] for pid in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x_vars[pid] * capital_dict[pid] for pid in project_ids)) <= 1000, name='budget')
    m.addConstr(x_vars[id_rnd_alpha] + x_vars[id_global_expansion] <= 1, name='mutual_exclusive_4_7')
    m.addConstr(x_vars[id_sys_auto] <= x_vars[id_infra_upgrade], name='prereq_6_requires_1')
    m.addConstr(x_vars[id_cust_exp_platform] <= x_vars[id_staff_training], name='contingent_10_requires_5')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_project_selection()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')