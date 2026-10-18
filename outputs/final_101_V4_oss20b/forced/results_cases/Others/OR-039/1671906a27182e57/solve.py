import gurobipy as gp
import pandas as pd
import re
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
warehouse_ids = df_capacity['Warehouse ID'].tolist()
capacities = df_capacity.set_index('Warehouse ID')['Capacity'].to_dict()
product_names = df_products['ProductName'].tolist()
values = df_products.set_index('ProductName')['Value'].to_dict()
weights = df_products.set_index('ProductName')['Weight'].to_dict()

def _short(name: str) -> str:
    return re.sub('\\W+', '_', name)
m = gp.Model('CarInventoryOptimization')
x = {}
for i in product_names:
    for w in warehouse_ids:
        var_name = f'x_{_short(i)}_{_short(w)}'
        x[i, w] = m.addVar(vtype=gp.GRB.INTEGER)
obj = gp.quicksum((values[i] * x[i, w] for i in product_names for w in warehouse_ids))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for w in warehouse_ids:
    cap_expr = gp.quicksum((weights[i] * x[i, w] for i in product_names))
    m.addConstr(cap_expr <= capacities[w])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Inventory plan ---')
    for w in warehouse_ids:
        print(f'Warehouse {w}:')
        for i in product_names:
            qty = x[i, w].X
            if qty > 0:
                print(f'  {i}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')