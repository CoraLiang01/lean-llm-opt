import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()
name_to_id = {n.casefold().strip(): pid for pid, n in zip(df['Project ID'], df['Project Name'])}

def get_project_id_by_name(target_name):
    key = target_name.casefold().strip()
    if key not in name_to_id:
        raise ValueError(f"Project name '{target_name}' not found in project.csv")
    return int(name_to_id[key])
id_rnd_alpha = get_project_id_by_name('R&D Initiative Alpha')
id_global_expansion = get_project_id_by_name('Global Expansion Pilot')
id_system_automation = get_project_id_by_name('System Automation')
id_infra_upgrade = get_project_id_by_name('Infrastructure Upgrade')
id_cust_exp_platform = get_project_id_by_name('Customer Experience Platform')
id_staff_training = get_project_id_by_name('Staff Training Program')
for pid in [id_rnd_alpha, id_global_expansion, id_system_automation, id_infra_upgrade, id_cust_exp_platform, id_staff_training]:
    if pid not in project_ids:
        raise ValueError(f'Project ID {pid} not found in project.csv')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[id_rnd_alpha] + x[id_global_expansion] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[id_system_automation] <= x[id_infra_upgrade], name='prereq_6_1')
m.addConstr(x[id_cust_exp_platform] <= x[id_staff_training], name='contingent_10_5')
m.optimize()