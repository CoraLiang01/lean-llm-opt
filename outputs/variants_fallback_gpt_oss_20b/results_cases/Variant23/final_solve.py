import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Path to the prepared JSON file
DATA_PATH = (
    ""
    "work/Variant23/prepared_inputs.json"
)

# Load the JSON and build a dictionary of tables keyed by their name
data_json = pd.read_json(DATA_PATH, typ="series")
tables = {t["name"]: t["rows"] for t in data_json["tables"]}

# Convert each table to a DataFrame
depot_df = pd.DataFrame(tables["depot_capacity"])
market_df = pd.DataFrame(tables["market_demand"])
varcost_df = pd.DataFrame(tables["route_variable_costs"])
fixedcost_df = pd.DataFrame(tables["route_fixed_costs"])

# Ensure numeric columns are floats
depot_df["SupplyCapacity"] = depot_df["SupplyCapacity"].astype(float)
market_df["Demand"] = market_df["Demand"].astype(float)
for col in ["M1", "M2", "M3", "M4", "M5"]:
    varcost_df[col] = varcost_df[col].astype(float)
    fixedcost_df[col] = fixedcost_df[col].astype(float)

# Sets
depots = depot_df["Depot"].tolist()
markets = market_df["Market"].tolist()

# Parameters
supply = dict(zip(depot_df["Depot"], depot_df["SupplyCapacity"]))
demand = dict(zip(market_df["Market"], market_df["Demand"]))

# Variable cost dictionary
c = {}
for _, row in varcost_df.iterrows():
    d = row["Depot"]
    for market in markets:
        key = (d, market)
        c[key] = float(row.get(market, 0.0))  # default 0.0 if missing

# Fixed cost dictionary
f = {}
for _, row in fixedcost_df.iterrows():
    d = row["Depot"]
    for market in markets:
        key = (d, market)
        f[key] = float(row.get(market, 0.0))  # default 0.0 if missing

# Big‑M values
M = {(d, m): min(supply[d], demand[m]) for d in depots for m in markets}

# Build the Gurobi model
model = gp.Model()
model.setParam("MIPGap", 1e-6)

# Decision variables
x = model.addVars(
    [(d, m) for d in depots for m in markets],
    vtype=GRB.CONTINUOUS,
    name="x",
)
y = model.addVars(
    [(d, m) for d in depots for m in markets],
    vtype=GRB.BINARY,
    name="y",
)

# Objective: variable shipping cost + fixed route activation cost
obj = gp.quicksum(c[(d, m)] * x[d, m] + f[(d, m)] * y[d, m] for d in depots for m in markets)
model.setObjective(obj, GRB.MINIMIZE)

# Market demand constraints
for market in markets:
    model.addConstr(
        gp.quicksum(x[d, market] for d in depots) == demand[market],
        name=f"Demand_{market}",
    )

# Depot supply constraints
for depot in depots:
    model.addConstr(
        gp.quicksum(x[depot, m] for m in markets) <= supply[depot],
        name=f"Supply_{depot}",
    )

# Linking constraints
for depot in depots:
    for market in markets:
        model.addConstr(
            x[depot, market] <= M[(depot, market)] * y[depot, market],
            name=f"Link_{depot}_{market}",
        )

# Optimize
model.optimize()

# Output results
if model.Status == GRB.OPTIMAL:
    print(f"Optimal objective value: {model.ObjVal}")
    for var in model.getVars():
        if var.X > 1e-6:
            print(f"{var.VarName} = {var.X}")
else:
    print(f"Solver status: {model.Status}")