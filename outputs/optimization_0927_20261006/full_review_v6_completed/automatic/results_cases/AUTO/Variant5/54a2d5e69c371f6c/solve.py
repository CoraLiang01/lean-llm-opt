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
machine_capacity_df['Machine'] = machine_capacity_df['Machine'].str.strip()
machine_capacity_df['Capacity'] = machine_capacity_df['Capacity'].astype(int)
capacity_dict = dict(zip(machine_capacity_df['Machine'], machine_capacity_df['Capacity']))
assignment_costs_df['Machine'] = assignment_costs_df['Machine'].str.strip()
cost_dict = {}
for (_, row) in assignment_costs_df.iterrows():
    machine = row['Machine']
    for job in jobs:
        cost_dict[machine, job] = int(row[job])
assignment_resources_df['Machine'] = assignment_resources_df['Machine'].str.strip()
resource_dict = {}
for (_, row) in assignment_resources_df.iterrows():
    machine = row['Machine']
    for job in jobs:
        resource_dict[machine, job] = int(row[job])
for machine in machines:
    if machine not in capacity_dict:
        raise ValueError(f'Missing capacity for machine {machine}')
    for job in jobs:
        if (machine, job) not in cost_dict:
            raise ValueError(f'Missing assignment cost for machine {machine}, job {job}')
        if (machine, job) not in resource_dict:
            raise ValueError(f'Missing assignment resource for machine {machine}, job {job}')
m = gp.Model('GeneralizedAssignment')
x_vars = m.addVars(machines, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost_dict[machine, job] * x_vars[machine, job] for machine in machines for job in jobs)), gp.GRB.MINIMIZE)
for job in jobs:
    m.addConstr(gp.quicksum((x_vars[machine, job] for machine in machines)) == 1, name=f'assign_{job}')
for machine in machines:
    m.addConstr(gp.quicksum((resource_dict[machine, job] * x_vars[machine, job] for job in jobs)) <= capacity_dict[machine], name=f'capacity_{machine}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.2f}')
    print('--- Assignment Plan ---')
    for job in jobs:
        for machine in machines:
            if x_vars[machine, job].X > 0.5:
                print(f'Job {job} assigned to Team {machine} (Cost: {cost_dict[machine, job]}, Resource: {resource_dict[machine, job]})')
else:
    print(f'No optimal solution found. Status: {m.status}')