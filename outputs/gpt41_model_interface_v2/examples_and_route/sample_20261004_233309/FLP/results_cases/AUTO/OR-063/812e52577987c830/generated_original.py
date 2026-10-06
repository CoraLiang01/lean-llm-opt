```python
import gurobipy as gp
import pandas as pd
import numpy as np

# 1. Read data from CSVs

# Read demand.csv: customer, demand
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv', sep=',')
# Normalize customer IDs as strings, strip whitespace
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = demand_df['customer'].tolist()
demand = dict(zip(demand_df['customer'], demand_df['demand']))

# Read fixed_cost.csv: Unnamed: 0 (warehouse), fixed_costs
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv', sep=',')
fixed_cost_df['warehouse'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
warehouses = fixed_cost_df['warehouse'].tolist()
fixed_costs = dict(zip(fixed_cost_df['warehouse'], fixed_cost_df['fixed_costs']))

# Read transportation_costs.csv: Unnamed: 0 (warehouse), columns C1..C7
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', sep=',')
trans_costs_df['warehouse'] = trans_costs_df['Unnamed: 0'].astype(str).str.strip()

# Validate that warehouse and customer sets match across files
if set(warehouses) != set(trans_costs_df['warehouse']):
    raise ValueError("Mismatch between warehouses in fixed_cost.csv and transportation_costs.csv")
if set(customers) != set([c for c in trans_costs_df.columns if c.startswith('C')]):
    raise ValueError("Mismatch between customers in demand.csv and transportation_costs.csv")

# Build transportation cost dictionary: (warehouse, customer) -> cost
transportation_costs = {}
for _, row in trans_costs_df.iterrows():
    w = str(row['warehouse']).strip()
    for c in customers:
        transportation_costs[(w, c)] = float(row[c])

# 2. Build Gurobi model
m = gp.Model("UFLP_Bandcamp")

# 3. Decision variables
# y[i]: binary, 1 if warehouse i is open
y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='y')
# x[i,j]: amount supplied from warehouse i to customer j
x = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='x')

# 4. Objective: Minimize total cost (fixed + transportation)
fixed_cost_term = gp.quicksum(fixed_costs[i] * y[i] for i in warehouses)
transport_cost_term = gp.quicksum(transportation_costs[(i, j)] * x[i, j] for i in warehouses for j in customers)
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)

# 5. Constraints

# Demand satisfaction: for each customer, sum of supplies from all warehouses equals demand
for j in customers:
    m.addConstr(gp.quicksum(x[i, j] for i in warehouses) == demand[j], name=f"demand_{j}")

# Linking: x[i,j] <= demand[j] * y[i] for all i, j
for i in warehouses:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i], name=f"link_{i}_{j}")

# 6. Optimize
m.optimize()

# 7. Output results
if m.status == gp.GRB.OPTIMAL:
    print(f"Optimal total value/cost: {m.objVal:.2f}")
    print("\n--- Warehouse Activation ---")
    for i in warehouses:
        if y[i].X > 0.5:
            print(f"  Warehouse {i}: OPEN (fixed cost: {fixed_costs[i]:.2f})")
        else:
            print(f"  Warehouse {i}: CLOSED")
    print("\n--- Customer Assignments ---")
    for j in customers:
        print(f"Customer {j} (demand: {demand[j]}):")
        for i in warehouses:
            if x[i, j].X > 1e-6:
                print(f"  Supplied from {i}: {x[i, j].X:.2f} units (cost per unit: {transportation_costs[(i, j)]:.2f})")
else:
    print(f"No optimal solution found. Status: {m.status}")
```