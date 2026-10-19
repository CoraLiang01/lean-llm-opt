import gurobipy as gp
import pandas as pd
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
shelves = capacity_df['BookshelfID'].tolist()
products = products_df['ProductName'].tolist()
capacity_dict = dict(zip(capacity_df['BookshelfID'], capacity_df['Capacity']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
m = gp.Model('BookShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    for i in shelves:
        total_value = sum((value_dict[j] * x[i, j].X for j in products))
        total_weight = sum((weight_dict[j] * x[i, j].X for j in products))
        print(f'Shelf {i}: Value={total_value:.2f}, Weight={total_weight:.2f} / Capacity={capacity_dict[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')