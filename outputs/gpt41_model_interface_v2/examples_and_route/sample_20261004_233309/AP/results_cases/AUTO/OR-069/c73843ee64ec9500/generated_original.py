```python
import gurobipy as gp
import pandas as pd
import numpy as np
import re

# Load the cost matrix from CSV
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv', sep=',')

# Extract manager and project identifiers
managers = costs_df['Manager'].astype(str).tolist()

# Identify project columns (those matching 'Project X Cost')
project_col_pattern = re.compile(r'^Project\s+(\d+)\s+Cost$', re.IGNORECASE)
project_cols = [col for col in costs_df.columns if project_col_pattern.match(col)]
# Sort project columns by project number to preserve order
project_cols_sorted = sorted(
    project_cols,
    key=lambda c: int(project_col_pattern.match(c).group(1))
)
projects = [f'Project {project_col_pattern.match(col).group(1)}' for col in project_cols_sorted]

# Build cost dictionary: cost[manager][project]
cost = {}
for idx, row in costs_df.iterrows():
    manager = str(row['Manager'])
    cost[manager] = {}
    for col, project in zip(project_cols_sorted, projects):
        val = row[col]
        if pd.isnull(val):
            raise ValueError(f"Missing cost for manager '{manager}', project '{project}'")
        cost[manager][project] = float(val)

# Validate dimensions
if len(managers) != len(projects):
    raise ValueError(f"Number of managers ({len(managers)}) does not match number of projects ({len(projects)}).")

# Build Gurobi model
m = gp.Model("ManagerProjectAssignment")

# Decision variables: x[manager, project] = 1 if manager assigned to project, else 0
x = m.addVars(
    managers, projects,
    vtype=gp.GRB.BINARY,
    name=''
)

# Objective: minimize total assignment cost
m.setObjective(
    gp.quicksum(cost[manager][project] * x[manager, project] for manager in managers for project in projects),
    gp.GRB.MINIMIZE
)

# Constraint 1: Each manager assigned to exactly one project
for manager in managers:
    m.addConstr(
        gp.quicksum(x[manager, project] for project in projects) == 1,
        name=''
    )

# Constraint 2: Each project assigned to exactly one manager
for project in projects:
    m.addConstr(
        gp.quicksum(x[manager, project] for manager in managers) == 1,
        name=''
    )

# Optimize
m.optimize()

# Output results
if m.status == gp.GRB.OPTIMAL:
    print(f"Optimal total assignment cost: {m.objVal:.2f}")
    print("--- Assignment ---")
    for manager in managers:
        for project in projects:
            if x[manager, project].X > 0.5:
                print(f"{manager} assigned to {project} (Cost: {cost[manager][project]:.0f})")
else:
    print(f"No optimal solution found. Status: {m.status}")
```