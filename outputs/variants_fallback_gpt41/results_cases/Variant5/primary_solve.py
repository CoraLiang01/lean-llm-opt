import gurobipy as gp
import pandas as pd
import numpy as np
import re
machine_capacity_path = 'benchmark_dataset/Variant5/inputs/machine_capacity.csv'
assignment_costs_path = 'benchmark_dataset/Variant5/inputs/assignment_costs.csv'
assignment_resources_path = 'benchmark_dataset/Variant5/inputs/assignment_resources.csv'
machine_capacity_df = pd.read_csv(machine_capacity_path, dtype=str, keep_default_na=False)
if 'Machine' not in machine_capacity_df.columns or 'Capacity' not in machine_capacity_df.columns:
    raise KeyError("machine_capacity.csv must contain 'Machine' and 'Capacity' columns.")
machine_capacity_df['Machine'] = machine_capacity_df['Machine'].str.strip()
machines = list(machine_capacity_df['Machine'])
capacity_dict = {}
for (idx, row) in machine_capacity_df.iterrows():
    machine = row['Machine'].strip()
    try:
        capacity = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity value for machine '{machine}': {row['Capacity']}")
    capacity_dict[machine] = capacity
assignment_costs_df = pd.read_csv(assignment_costs_path, dtype=str, keep_default_na=False)
if 'Machine' not in assignment_costs_df.columns:
    raise KeyError("assignment_costs.csv must contain 'Machine' column.")
assignment_costs_df['Machine'] = assignment_costs_df['Machine'].str.strip()
job_columns = [col for col in assignment_costs_df.columns if col != 'Machine']
jobs = job_columns
expected_jobs = [f'J{i}' for i in range(1, 9)]
if sorted(jobs) != sorted(expected_jobs):
    raise ValueError(f'Jobs in assignment_costs.csv do not match expected jobs {expected_jobs}. Found: {jobs}')
cost_dict = {}
for (idx, row) in assignment_costs_df.iterrows():
    machine = row['Machine'].strip()
    for job in jobs:
        try:
            cost = int(row[job])
        except Exception:
            raise ValueError(f"Invalid cost value for machine '{machine}', job '{job}': {row[job]}")
        cost_dict[machine, job] = cost
assignment_resources_df = pd.read_csv(assignment_resources_path, dtype=str, keep_default_na=False)
if 'Machine' not in assignment_resources_df.columns:
    raise KeyError("assignment_resources.csv must contain 'Machine' column.")
assignment_resources_df['Machine'] = assignment_resources_df['Machine'].str.strip()
resource_job_columns = [col for col in assignment_resources_df.columns if col != 'Machine']
if sorted(resource_job_columns) != sorted(jobs):
    raise ValueError('Job columns in assignment_resources.csv do not match those in assignment_costs.csv.')
resource_dict = {}
for (idx, row) in assignment_resources_df.iterrows():
    machine = row['Machine'].strip()
    for job in jobs:
        try:
            resource = int(row[job])
        except Exception:
            raise ValueError(f"Invalid resource value for machine '{machine}', job '{job}': {row[job]}")
        resource_dict[machine, job] = resource
for machine in machines:
    for job in jobs:
        if (machine, job) not in cost_dict:
            raise KeyError(f"Missing cost for machine '{machine}', job '{job}'.")
        if (machine, job) not in resource_dict:
            raise KeyError(f"Missing resource for machine '{machine}', job '{job}'.")
m = gp.Model('GeneralizedAssignment')
x_vars = m.addVars(machines, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost_dict[i, j] * x_vars[i, j] for i in machines for j in jobs)), gp.GRB.MINIMIZE)
for j in jobs:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in machines)) == 1, name=f'assign_{j}')
for i in machines:
    m.addConstr(gp.quicksum((resource_dict[i, j] * x_vars[i, j] for j in jobs)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()