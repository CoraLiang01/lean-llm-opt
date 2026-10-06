```python
import gurobipy as gp
import pandas as pd
import numpy as np
import re

# 1. Load data
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'

# Read customer demand
df_demand = pd.read_csv(customer_demand_path, sep=',')
# Normalize customer IDs (strip whitespace, preserve case and leading zeros)
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))

# Read supply capacity
df_supply = pd.read_csv(supply_capacity_path, sep=',')
# The store ID is in 'Unnamed: 0'
df_supply['store'] = df_supply['Unnamed: 0'].astype(str).str.strip()
stores = df_supply['store'].tolist()
supply_capacity = dict(zip(df_supply['store'], df_supply['supply_capacity']))

# Read transportation costs
df_cost = pd.read_csv(transportation_costs_path, sep=',')
# Store IDs in 'Unnamed: 0', customer columns are C1, ..., C12
df_cost['store'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_matrix = df_cost.set_index('store')
# Ensure all required stores and customers are present
missing_stores = set(stores) - set(cost_matrix.index)
missing_customers = set(customers) - set(cost_matrix.columns)
if missing_stores:
    raise ValueError(f"Missing stores in transportation_costs.csv: {missing_stores}")
if missing_customers:
    raise ValueError(f"Missing customers in transportation_costs.csv: {missing_customers}")

# 2. Build Gurobi model
m = gp.Model("Walmart_Transportation")

# 3. Define variables: x[s, c] >= 0, continuous
x = m.addVars(
    stores, customers,
    lb=0.0,
    vtype=gp.GRB.CONTINUOUS,
    name=''
)

# 4. Set objective: Minimize total transportation cost
m.setObjective(
    gp.quicksum(
        cost_matrix.loc[s, c] * x[s, c]
        for s in stores for c in customers
    ),
    gp.GRB.MINIMIZE
)

# 5. Add constraints

# Demand satisfaction: For each customer, sum over stores == demand
for c in customers:
    m.addConstr(
        gp.quicksum(x[s, c] for s in stores) == demand[c],
        name=''
    )

# Supply capacity: For each store, sum over customers <= supply_capacity
for s in stores:
    m.addConstr(
        gp.quicksum(x[s, c] for c in customers) <= supply_capacity[s],
        name=''
    )

# 6. Optimize
m.optimize()

# 7. Print results
if m.status == gp.GRB.OPTIMAL:
    print(f"Optimal total transportation cost: {m.objVal:.6f}")
    print("Optimal transportation plan (quantities shipped from each store to each customer):")
    for s in stores:
        for c in customers:
            val = x[s, c].X
            if val > 1e-6:
                print(f"  Store {s} -> Customer {c}: {val:.6f}")
else:
    print(f"No optimal solution found. Status: {m.status}")
```