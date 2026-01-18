import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
pop_path = "/mnt/data/neighborhoods_population.csv"
dist_path = "/mnt/data/distance.csv"
cap_path = "/mnt/data/school_capacity.csv"

# Read data
pop = pd.read_csv(pop_path)
cap = pd.read_csv(cap_path)
dist = pd.read_csv(dist_path, index_col=0)

# Prepare sets
neighborhoods = list(dist.columns)  # Adaptive neighborhood count
schools = list(dist.index)          # Adaptive school count

# Align and validate
pop = pop.set_index("Neighborhood").loc[neighborhoods]
cap = cap.set_index("School").loc[schools]

# Build model
model = gp.Model("School_Assignment_Scaled")

# Decision variables: x_w[i,j], x_n[i,j] (integer non-negative)
x_w, x_n = {}, {}
for school in schools:
    for neigh in neighborhoods:
        x_w[(school, neigh)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_w_{school}_{neigh}")
        x_n[(school, neigh)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_n_{school}_{neigh}")

model.update()

# Objective: minimize total "student-miles"
model.setObjective(
    gp.quicksum(dist.loc[school, neigh] * (x_w[(school, neigh)] + x_n[(school, neigh)])
                for school in schools for neigh in neighborhoods),
    GRB.MINIMIZE
)

# Neighborhood supply conservation (white/non-white separately)
for neigh in neighborhoods:
    model.addConstr(gp.quicksum(x_w[(s, neigh)] for s in schools) == int(pop.loc[neigh, "Population_White"]),
                    name=f"white_supply[{neigh}]")
    model.addConstr(gp.quicksum(x_n[(s, neigh)] for s in schools) == int(pop.loc[neigh, "Population_NonWhite"]),
                    name=f"nonwhite_supply[{neigh}]")

# School capacity
for school in schools:
    model.addConstr(
        gp.quicksum(x_w[(school, neigh)] + x_n[(school, neigh)] for neigh in neighborhoods)
        <= int(cap.loc[school, "Capacity"]),
        name=f"capacity[{school}]"
    )

# School-level "racial balance": 60%±10 percentage points → [0.5, 0.7]
for school in schools:
    total = gp.quicksum(x_w[(school, neigh)] + x_n[(school, neigh)] for neigh in neighborhoods)
    white = gp.quicksum(x_w[(school, neigh)] for neigh in neighborhoods)
    model.addConstr(white >= 0.50 * total, name=f"white_min[{school}]")
    model.addConstr(white <= 0.70 * total, name=f"white_max[{school}]")

# Scale information
model.update()
print(f"|S|={len(schools)}, |N|={len(neighborhoods)} => variables = 2*|S|*|N| = {2*len(schools)*len(neighborhoods)}")
print(f"Gurobi counted variables: {model.NumVars}, constraints: {model.NumConstrs}")

# Optimize
model.optimize()

# Write LP file
model.write(os.path.join(script_dir, "model.lp"))

# Output
if model.status == GRB.OPTIMAL:
    print(f"\nOptimal objective (student-miles): {model.ObjVal:.2f}\n")
    for school in schools:
        w_i = sum(x_w[(school, n)].X for n in neighborhoods)
        n_i = sum(x_n[(school, n)].X for n in neighborhoods)
        tot = w_i + n_i
        pct_w = (w_i / tot * 100) if tot > 0 else 0.0
        print(f"[{school}] Total={tot:.0f} (White={w_i:.0f}, {pct_w:.1f}%; Nonwhite={n_i:.0f}, {100-pct_w:.1f}%)")
        for neigh in neighborhoods:
            w = x_w[(school, neigh)].X
            n = x_n[(school, neigh)].X
            if w + n > 0:
                print(f"  <- {neigh}: White={w:.0f}, Nonwhite={n:.0f}")
else:
    print(f"Model not optimal. Status={model.status}")
