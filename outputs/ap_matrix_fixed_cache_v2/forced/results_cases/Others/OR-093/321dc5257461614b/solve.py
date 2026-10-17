import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
machine_ids = cost_df['Machine'].astype(str).str.strip().tolist()
task_ids = [col for col in cost_df.columns if col != 'Machine']
cost_dict = {}
for idx, row in cost_df.iterrows():
    machine = str(row['Machine']).strip()
    for task in task_ids:
        try:
            cost_val = float(row[task])
        except ValueError:
            raise ValueError(f"Invalid cost value for machine '{machine}', task '{task}': {row[task]}")
        cost_dict[machine, task] = cost_val
if len(machine_ids) != len(task_ids):
    raise ValueError(f'Number of machines ({len(machine_ids)}) does not match number of tasks ({len(task_ids)}).')
if len(machine_ids) != 12 or len(task_ids) != 12:
    raise ValueError('Expected 12 machines and 12 tasks as per the problem statement.')
m = gp.Model('AssignmentProblem')
x_vars = m.addVars(machine_ids, task_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost_dict[m_id, t_id] * x_vars[m_id, t_id] for m_id in machine_ids for t_id in task_ids)), gp.GRB.MINIMIZE)
for m_id in machine_ids:
    m.addConstr(gp.quicksum((x_vars[m_id, t_id] for t_id in task_ids)) == 1, name=f'assign_machine_{m_id}')
for t_id in task_ids:
    m.addConstr(gp.quicksum((x_vars[m_id, t_id] for m_id in machine_ids)) == 1, name=f'assign_task_{t_id}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for m_id in machine_ids:
        for t_id in task_ids:
            if x_vars[m_id, t_id].X > 0.5:
                print(f'Machine {m_id} assigned to Task {t_id} with cost {cost_dict[m_id, t_id]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')