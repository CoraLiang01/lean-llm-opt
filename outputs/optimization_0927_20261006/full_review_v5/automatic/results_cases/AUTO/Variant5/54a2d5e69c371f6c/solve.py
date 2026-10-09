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
jobs = [col.strip() for col in job_columns]
machine_capacity_df['Capacity'] = machine_capacity_df['Capacity'].astype(int)
capacity_dict = dict(zip(machine_capacity_df['Machine'].str.strip(), machine_capacity_df['Capacity']))
cost_dict = {}
for (_, row) in assignment_costs_df.iterrows():
    machine = row['Machine'].strip()
    for job in jobs:
        val = row[job]
        try:
            cost_dict[machine, job] = int(val)
        except Exception:
            raise ValueError(f'Invalid assignment cost for machine {machine}, job {job}: {val}')
resource_dict = {}
for (_, row) in assignment_resources_df.iterrows():
    machine = row['Machine'].strip()
    for job in jobs:
        val = row[job]
        try:
            resource_dict[machine, job] = int(val)
        except Exception:
            raise ValueError(f'Invalid assignment resource for machine {machine}, job {job}: {val}')
for machine in machines:
    if machine not in capacity_dict:
        raise KeyError(f'Missing capacity for machine {machine}')
    for job in jobs:
        if (machine, job) not in cost_dict:
            raise KeyError(f'Missing assignment cost for machine {machine}, job {job}')
        if (machine, job) not in resource_dict:
            raise KeyError(f'Missing assignment resource for machine {machine}, job {job}')
m = gp.Model('GeneralizedAssignment')
x_vars = m.addVars(machines, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost_dict[i, j] * x_vars[i, j] for i in machines for j in jobs)), gp.GRB.MINIMIZE)
for j in jobs:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in machines)) == 1, name=f'assign_{j}')
for i in machines:
    m.addConstr(gp.quicksum((resource_dict[i, j] * x_vars[i, j] for j in jobs)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.0f}')
    print('--- Assignment ---')
    for j in jobs:
        for i in machines:
            if x_vars[i, j].X > 0.5:
                print(f'Job {j} assigned to Team {i} (Cost: {cost_dict[i, j]}, Resource: {resource_dict[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')