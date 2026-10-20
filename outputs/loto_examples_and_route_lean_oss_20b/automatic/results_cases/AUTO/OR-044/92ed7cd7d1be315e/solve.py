import gurobipy as gp
import pandas as pd
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
capacity_df = pd.read_csv(capacity_path)
products_df = pd.read_csv(products_path)
sections = capacity_df['SectionID'].tolist()
capacities = dict(zip(capacity_df['SectionID'], capacity_df['Capacity']))
products = products_df['ProductName'].tolist()
values = dict(zip(products_df['ProductName'], products_df['Value']))
weights = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('SupermarketStocking')
x = m.addVars(sections, products, vtype=gp.GRB.INTEGER)
obj = gp.quicksum((values[j] * x[i, j] for i in sections for j in products))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for i in sections:
    m.addConstr(gp.quicksum((weights[j] * x[i, j] for j in products)) <= capacities[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal}')
    for i in sections:
        for j in products:
            if x[i, j].X > 0:
                print(f'Section {i}, Product {j}: {int(x[i, j].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')