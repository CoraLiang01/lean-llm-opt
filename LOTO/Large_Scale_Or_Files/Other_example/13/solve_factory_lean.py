import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Load data
# The CSV file should have columns:
# Product, Profit_A, Profit_B, Labor_A, Labor_B, Machine_A, Machine_B, RawMat_A, RawMat_B
df = pd.read_csv('full_data.csv').set_index('Product')

products = df.index.tolist()

# Model
m = gp.Model("TwoFactory_ProfitMax")

# Decision variables: x_A_p and x_B_p for all products
x_A = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name="x_A")
x_B = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name="x_B")

# Objective function
m.setObjective(
    gp.quicksum(df.loc[p, 'Profit_A'] * x_A[p] for p in products) +
    gp.quicksum(df.loc[p, 'Profit_B'] * x_B[p] for p in products),
    GRB.MAXIMIZE
)

# Factory A constraints
m.addConstr(
    gp.quicksum(df.loc[p, 'Labor_A'] * x_A[p] for p in products) <= 100,
    name="Labor_A"
)
m.addConstr(
    gp.quicksum(df.loc[p, 'Machine_A'] * x_A[p] for p in products) <= 90,
    name="Machine_A"
)
m.addConstr(
    gp.quicksum(df.loc[p, 'RawMat_A'] * x_A[p] for p in products) <= 80,
    name="RawMat_A"
)

# Factory B constraints
m.addConstr(
    gp.quicksum(df.loc[p, 'Labor_B'] * x_B[p] for p in products) <= 80,
    name="Labor_B"
)
m.addConstr(
    gp.quicksum(df.loc[p, 'Machine_B'] * x_B[p] for p in products) <= 100,
    name="Machine_B"
)
m.addConstr(
    gp.quicksum(df.loc[p, 'RawMat_B'] * x_B[p] for p in products) <= 90,
    name="RawMat_B"
)

# Optimize
m.optimize()

# Output results
if m.status == GRB.OPTIMAL:
    print(f"Optimal total profit: {m.objVal:.2f}")
    for p in products:
        if x_A[p].X > 1e-6 or x_B[p].X > 1e-6:
            print(f"{p}: Factory A = {x_A[p].X:.2f}, Factory B = {x_B[p].X:.2f}")
