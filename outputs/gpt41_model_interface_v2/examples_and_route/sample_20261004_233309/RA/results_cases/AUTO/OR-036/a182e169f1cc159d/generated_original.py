```python
import gurobipy as gp
import pandas as pd
import numpy as np

# 1. Read data
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'

# Read products.csv: all rows required
products_df = pd.read_csv(products_path, sep=',')
# Read capacity.csv: single value
capacity_df = pd.read_csv(capacity_path, sep=',')

# 2. Prepare index sets and parameters
# Index set: all ProductName in products.csv
product_ids = products_df['ProductName'].astype(str).tolist()

# Parameters: Value and Weight per product
# Ensure mapping is by ProductName (string)
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))

# Capacity: single integer value
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])

# 3. Build Gurobi model
m = gp.Model('CarSalesInventoryReplenishment')

# 4. Decision variables: x[i] = integer number of units to order for product i (>=0)
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='x')

# 5. Objective: maximize total benefit
m.setObjective(gp.quicksum(value_dict[i] * x[i] for i in product_ids), gp.GRB.MAXIMIZE)

# 6. Constraints

# Inventory capacity: sum_i Weight[i] * x[i] <= Capacity
m.addConstr(gp.quicksum(weight_dict[i] * x[i] for i in product_ids) <= capacity, name='inventory_capacity')

# 7. Optimize
m.optimize()

# 8. Output results
if m.status == gp.GRB.OPTIMAL:
    print(f"Optimal total value/cost: {m.objVal:.2f}")
    print("--- Optimal Daily Ordering Plan ---")
    for i in product_ids:
        xi = x[i].X
        if xi > 1e-6:
            print(f"  {i}: {int(round(xi))} units")
else:
    print(f"No optimal solution found. Status: {m.status}")
```