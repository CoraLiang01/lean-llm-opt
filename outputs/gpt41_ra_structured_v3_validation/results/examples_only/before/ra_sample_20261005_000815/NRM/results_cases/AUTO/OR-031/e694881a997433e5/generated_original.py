from gurobipy import Model, GRB

# Extract data from CSVQA_DATA
data_table = None
for t in CSVQA_DATA["tables"]:
    if t["table_id"] == "file_0_view_0":
        data_table = t
        break
if data_table is None:
    raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")

products = []
D = {}
I = {}
R = {}

for rec in data_table["records"]:
    v = rec["values"]
    prod = v["Full_Product_Name"]
    try:
        demand = int(v["Demand"])
        inventory = int(v["Initial Inventory"])
        revenue = float(v["Revenue"])
    except Exception as e:
        raise RuntimeError(f"Invalid data for product {prod}: {e}")
    products.append(prod)
    D[prod] = demand
    I[prod] = inventory
    R[prod] = revenue

# Validate all products have required data
for prod in products:
    if prod not in D or prod not in I or prod not in R:
        raise RuntimeError(f"Missing data for product {prod}")

# Build model
m = Model()
m.setParam("MIPGap", 1e-4)

# Decision variables: x_i (integer, 0 <= x_i <= min{D_i, I_i})
x = m.addVars(products, vtype=GRB.INTEGER, lb=0,
              ub={i: min(D[i], I[i]) for i in products}, name='x', name='')

# Objective: maximize total revenue
m.setObjective(sum(R[i] * x[i] for i in products), GRB.MAXIMIZE)

# Constraints: x_i <= I_i, x_i <= D_i, x_i >= 0 (already handled by bounds)
for i in products:
    m.addConstr(x[i] <= I[i], name='')
    m.addConstr(x[i] <= D[i], name='')

m.optimize()

if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for i in products:
        print(x[i].VarName, x[i].X)
else:
    print(m.Status)