import gurobipy as gp
import pandas as pd
capacity_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv')
products_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv')
shelves = capacity_df['ShelfID'].tolist()
capacity = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
products = products_df['ProductName'].tolist()
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('BigMartShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.CONTINUOUS, lb=0)
obj = gp.quicksum((value[j] * x[i, j] for i in shelves for j in products))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    for i in shelves:
        total_weight = sum((weight[j] * x[i, j].X for j in products))
        total_value = sum((value[j] * x[i, j].X for j in products))
        print(f'Shelf {i}: used weight {total_weight:.2f} / capacity {capacity[i]}, value {total_value:.2f}')
        for j in products:
            if x[i, j].X > 1e-06:
                print(f'  Product {j}: {x[i, j].X:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')