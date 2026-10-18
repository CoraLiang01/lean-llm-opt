import gurobipy as gp
import pandas as pd
import numpy as np
machine_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/machine_capacity.csv'
assignment_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_costs.csv'
assignment_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_resources.csv'
machine_capacity_df = pd.read_csv(machine_capacity_path, dtype=str, keep_default_na=False)
assignment_costs_df = pd.read_csv(assignment_costs_path, dtype=str, keep_default_na=False)
assignment_resources_df = pd.read_csv(assignment_resources_path, dtype=str, keep_default_na=False)
machines = machine_capacity_df['Machine'].str.strip().tolist()
job_columns = [col for col in assignment_costs_df.columns if col != 'Machine']
jobs = job_columns
capacity = {}
for idx, row in machine_capacity_df.iterrows():
    machine = row['Machine'].strip()
    try:
        cap = int(row['Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid capacity for machine '{machine}': {row['Capacity']}")
    capacity[machine] = cap
cost = {}
for idx, row in assignment_costs_df.iterrows():
    machine = row['Machine'].strip()
    for job in jobs:
        try:
            cost_val = int(row[job])
        except Exception as e:
            raise ValueError(f"Invalid cost for machine '{machine}', job '{job}': {row[job]}")
        cost[machine, job] = cost_val
resource = {}
for idx, row in assignment_resources_df.iterrows():
    machine = row['Machine'].strip()
    for job in jobs:
        try:
            res_val = int(row[job])
        except Exception as e:
            raise ValueError(f"Invalid resource for machine '{machine}', job '{job}': {row[job]}")
        resource[machine, job] = res_val
for machine in machines:
    if machine not in capacity:
        raise KeyError(f"Machine '{machine}' missing from capacity data.")
    for job in jobs:
        if (machine, job) not in cost:
            raise KeyError(f"Missing cost for machine '{machine}', job '{job}'.")
        if (machine, job) not in resource:
            raise KeyError(f"Missing resource for machine '{machine}', job '{job}'.")
m = gp.Model('GeneralizedAssignment')
x_vars = m.addVars(machines, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x_vars[i, j] for i in machines for j in jobs)), gp.GRB.MINIMIZE)
for j in jobs:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in machines)) == 1, name=f'assign_{j}')
for i in machines:
    m.addConstr(gp.quicksum((resource[i, j] * x_vars[i, j] for j in jobs)) <= capacity[i], name=f'cap_{i}')
m.optimize()