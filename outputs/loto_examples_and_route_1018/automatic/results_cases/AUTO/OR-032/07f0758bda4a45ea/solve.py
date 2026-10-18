LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_15.15","Revenue":"15.15","Demand":"1980","Initial Inventory":"9920.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_30.3","Revenue":"30.3","Demand":"3024","Initial Inventory":"20160.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_45.45","Revenue":"45.45","Demand":"4536","Initial Inventory":"30000.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_60.6","Revenue":"60.6","Demand":"5601","Initial Inventory":"38360.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_75.75","Revenue":"75.75","Demand":"7567","Initial Inventory":"51450.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_15.15', 'Revenue': '15.15', 'Demand': '1980', 'Initial Inventory': '9920.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_30.3', 'Revenue': '30.3', 'Demand': '3024', 'Initial Inventory': '20160.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_45.45', 'Revenue': '45.45', 'Demand': '4536', 'Initial Inventory': '30000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_60.6', 'Revenue': '60.6', 'Demand': '5601', 'Initial Inventory': '38360.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_75.75', 'Revenue': '75.75', 'Demand': '7567', 'Initial Inventory': '51450.0'}}]
import gurobipy as gp
from gurobipy import GRB
books_records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product_Name', '').startswith('Books_')]
books_ids = []
revenue = {}
demand = {}
inventory = {}
for rec in books_records:
    vals = rec['values']
    pid = vals['Product_Name']
    try:
        r = float(vals['Revenue'])
        d = int(float(vals['Demand']))
        s = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Invalid data for product {pid}: {e}')
    books_ids.append(pid)
    revenue[pid] = r
    demand[pid] = d
    inventory[pid] = s
if not set(books_ids) == set(revenue) == set(demand) == set(inventory):
    raise ValueError('Mismatch in product identifiers among parameters.')
m = gp.Model('Books_Revenue_Maximization')
books_vars = m.addVars(books_ids, lb=0, ub={pid: min(demand[pid], inventory[pid]) for pid in books_ids}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[pid] * books_vars[pid] for pid in books_ids)), GRB.MAXIMIZE)
m.addConstrs((books_vars[pid] <= demand[pid] for pid in books_ids), name='')
m.addConstrs((books_vars[pid] <= inventory[pid] for pid in books_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')