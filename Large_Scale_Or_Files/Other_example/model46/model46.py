import pandas as pd
import gurobipy as gp
from gurobipy import GRB

import os
CSV_PATH = os.path.join(os.path.dirname(__file__), "cost.csv")  # use absolute path to CSV in same folder
df = pd.read_csv(CSV_PATH)

foods = df["Food"].tolist()
cal   = dict(zip(df["Food"], df["Calories"]))
prot  = dict(zip(df["Food"], df["Protein"]))
fat   = dict(zip(df["Food"], df["Fat"]))
vitc  = dict(zip(df["Food"], df["VitaminC"]))
cost  = dict(zip(df["Food"], df["Cost"]))

CAL_MIN, PROT_MIN, VITC_MIN, FAT_MAX = 2000, 50, 60, 70

m = gp.Model("diet_lp_120")
x = m.addVars(foods, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
m.setObjective(gp.quicksum(cost[f] * x[f] for f in foods), GRB.MINIMIZE)

m.addConstr(gp.quicksum(cal[f]  * x[f] for f in foods) >= CAL_MIN,  name="calories")
m.addConstr(gp.quicksum(prot[f] * x[f] for f in foods) >= PROT_MIN, name="protein")
m.addConstr(gp.quicksum(vitc[f] * x[f] for f in foods) >= VITC_MIN, name="vitaminC")
m.addConstr(gp.quicksum(fat[f]  * x[f] for f in foods) <= FAT_MAX,  name="fat")

m.Params.OutputFlag = 1
m.optimize()

if m.status == GRB.OPTIMAL:
    sol = []
    for f in foods:
        val = x[f].X
        if abs(val) > 1e-8:
            sol.append({
                "Food": f,
                "Servings": val,
                "Cal": cal[f] * val,
                "Protein": prot[f] * val,
                "Fat": fat[f] * val,
                "VitC": vitc[f] * val,
                "Cost": cost[f] * val
            })
    sol_df = pd.DataFrame(sol).sort_values("Servings", ascending=False)
    print("\nTop items in the optimal plan (by servings):")
    print(sol_df.head(20).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nTotal cost = ${m.objVal:.2f}")
    print("\nFull solution written to solution_120.csv")
else:
    print("Model not solved to optimality. Status:", m.status)