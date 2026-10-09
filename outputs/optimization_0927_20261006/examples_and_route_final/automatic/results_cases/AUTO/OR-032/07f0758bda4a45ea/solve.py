LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_15.15","Revenue":"15.15","Demand":"1980","Initial Inventory":"9920.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_30.3","Revenue":"30.3","Demand":"3024","Initial Inventory":"20160.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_45.45","Revenue":"45.45","Demand":"4536","Initial Inventory":"30000.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_60.6","Revenue":"60.6","Demand":"5601","Initial Inventory":"38360.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_75.75","Revenue":"75.75","Demand":"7567","Initial Inventory":"51450.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_15.15', 'Revenue': '15.15', 'Demand': '1980', 'Initial Inventory': '9920.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_30.3', 'Revenue': '30.3', 'Demand': '3024', 'Initial Inventory': '20160.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_45.45', 'Revenue': '45.45', 'Demand': '4536', 'Initial Inventory': '30000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_60.6', 'Revenue': '60.6', 'Demand': '5601', 'Initial Inventory': '38360.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_75.75', 'Revenue': '75.75', 'Demand': '7567', 'Initial Inventory': '51450.0'}}]
import gurobipy as gp
from gurobipy import GRB
books_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product_Name', '').startswith('Books_')]
products = []
revenue = {}
demand = {}
inventory = {}
for rec in books_records:
    vals = rec['values']
    pid = vals['Product_Name']
    products.append(pid)
    try:
        revenue[pid] = float(vals['Revenue'])
        demand[pid] = int(float(vals['Demand']))
        inventory[pid] = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Error parsing record for {pid}: {e}')
for pid in products:
    if pid not in revenue or pid not in demand or pid not in inventory:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('Books_Fulfillment')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= demand[pid] for pid in products), name='')
m.addConstrs((x_vars[pid] <= inventory[pid] for pid in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')