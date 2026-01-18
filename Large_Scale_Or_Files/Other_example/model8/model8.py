# model8.py
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# === Read CSV files ===
fac_df = pd.read_csv(os.path.join(script_dir, "facility_costs.csv"))        # Facility, FixedCost, Capacity
dem_df = pd.read_csv(os.path.join(script_dir, "demand_requirements.csv"))   # Destination, Demand
ship_df = pd.read_csv(os.path.join(script_dir, "shipping_costs.csv"))       # Origin, B1, B2, ..., B8

facility_names = fac_df["Facility"].tolist()
customer_names  = dem_df["Destination"].tolist()

fixed_costs = fac_df.set_index("Facility")["FixedCost"].reindex(facility_names).to_numpy()
capacities  = fac_df.set_index("Facility")["Capacity"].reindex(facility_names).to_numpy()
demands     = dem_df.set_index("Destination")["Demand"].reindex(customer_names).to_numpy()

# Align shipping cost matrix by (facility × customer)
ship_df = ship_df.set_index("Origin").reindex(facility_names)[customer_names]
shipping_costs = ship_df.to_numpy()

I, J = len(facility_names), len(customer_names)

# Quick consistency check
total_capacity = capacities.sum()
total_demand   = demands.sum()
if total_capacity < total_demand:
    raise ValueError(f"Total capacity {total_capacity} < Total demand {total_demand}, problem is infeasible.")

# === Modeling ===
m = gp.Model("Capacitated_Facility_Location")

# Decision variables
y = m.addVars(I, vtype=GRB.BINARY, name="y")
x = m.addVars(I, J, lb=0.0, name="x")

# To force A1 to be enabled: uncomment the next line
# y[facility_names.index("A1")].LB = 1

# Objective function: fixed cost + shipping cost
m.setObjective(
    gp.quicksum(fixed_costs[i] * y[i] for i in range(I)) +
    gp.quicksum(shipping_costs[i, j] * x[i, j] for i in range(I) for j in range(J)),
    GRB.MINIMIZE
)

# Demand constraints
for j in range(J):
    m.addConstr(gp.quicksum(x[i, j] for i in range(I)) == demands[j], name=f"demand[{customer_names[j]}]")

# Capacity-activation linkage
for i in range(I):
    m.addConstr(gp.quicksum(x[i, j] for j in range(J)) <= capacities[i] * y[i], name=f"cap[{facility_names[i]}]")

# Optional: make solver prefer fewer open facilities (heuristic)
# m.setParam(GRB.Param.MIPFocus, 1)

m.optimize()

# Export model
m.write(os.path.join(script_dir, "model.lp"))

# === Output results ===
if m.status == GRB.OPTIMAL:
    print(f"Optimal objective = {m.objVal:.2f}")
    open_sites = [facility_names[i] for i in range(I) if y[i].X > 0.5]
    print(f"Open facilities ({len(open_sites)}): {open_sites}")

    print("Shipments i->j (qty @ unit_cost):")
    for i in range(I):
        for j in range(J):
            q = x[i, j].X
            if q > 1e-6:
                print(f"  {facility_names[i]} -> {customer_names[j]} : {q:.0f} @ {shipping_costs[i, j]}")
else:
    print(f"Model status: {m.status}")

