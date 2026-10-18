import gurobipy as gp
import pandas as pd
capacity_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv')
capacity = capacity_df['Capacity'].iloc[0]
products_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv')
product_names = products_df['ProductName'].tolist()
weights = dict(zip(product_names, products_df['Weight']))
values = dict(zip(product_names, products_df['Value']))
m = gp.Model('SupermarketStockOptimization')
x = m.addVars(range(len(product_names)))
prod_to_var = {product_names[i]: x[i] for i in range(len(product_names))}
m.setObjective(gp.quicksum((values[p] * prod_to_var[p] for p in product_names)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * prod_to_var[p] for p in product_names)) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('Optimal ordering plan:')
    for p in product_names:
        qty = prod_to_var[p].X
        if qty > 1e-06:
            print(f'  {p}: {qty:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')