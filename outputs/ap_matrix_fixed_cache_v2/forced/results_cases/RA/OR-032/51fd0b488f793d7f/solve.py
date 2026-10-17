LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv", "values": {"Product_Name": "Books_15.15", "Revenue": "15.15", "Demand": "1980", "Initial Inventory": "9920.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv", "values": {"Product_Name": "Books_30.3", "Revenue": "30.3", "Demand": "3024", "Initial Inventory": "20160.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv", "values": {"Product_Name": "Books_45.45", "Revenue": "45.45", "Demand": "4536", "Initial Inventory": "30000.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv", "values": {"Product_Name": "Books_60.6", "Revenue": "60.6", "Demand": "5601", "Initial Inventory": "38360.0"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv", "values": {"Product_Name": "Books_75.75", "Revenue": "75.75", "Demand": "7567", "Initial Inventory": "51450.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_15.15', 'Revenue': '15.15', 'Demand': '1980', 'Initial Inventory': '9920.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_30.3', 'Revenue': '30.3', 'Demand': '3024', 'Initial Inventory': '20160.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_45.45', 'Revenue': '45.45', 'Demand': '4536', 'Initial Inventory': '30000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_60.6', 'Revenue': '60.6', 'Demand': '5601', 'Initial Inventory': '38360.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_75.75', 'Revenue': '75.75', 'Demand': '7567', 'Initial Inventory': '51450.0'}}]
import gurobipy as gp
from gurobipy import GRB
books_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product_Name', '').startswith('Books_')]
product_names = []
revenue = {}
demand = {}
inventory = {}
for rec in books_records:
    name = rec['values']['Product_Name']
    product_names.append(name)
    try:
        revenue[name] = float(rec['values']['Revenue'])
        demand[name] = int(float(rec['values']['Demand']))
        inventory[name] = int(float(rec['values']['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Invalid data for product {name}: {e}')
upper_bounds = {name: min(demand[name], inventory[name]) for name in product_names}
for name in product_names:
    if name not in revenue or name not in demand or name not in inventory:
        raise ValueError(f'Missing data for product {name}')
m = gp.Model('Books_Revenue_Maximization')
x_vars = m.addVars(product_names, lb=0, ub=[upper_bounds[name] for name in product_names], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[name] * x_vars[name] for name in product_names)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')