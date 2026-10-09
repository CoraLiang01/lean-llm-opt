LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_15.15","Revenue":"15.15","Demand":"1980","Initial Inventory":"9920.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_30.3","Revenue":"30.3","Demand":"3024","Initial Inventory":"20160.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_45.45","Revenue":"45.45","Demand":"4536","Initial Inventory":"30000.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_60.6","Revenue":"60.6","Demand":"5601","Initial Inventory":"38360.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_75.75","Revenue":"75.75","Demand":"7567","Initial Inventory":"51450.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_15.15', 'Revenue': '15.15', 'Demand': '1980', 'Initial Inventory': '9920.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_30.3', 'Revenue': '30.3', 'Demand': '3024', 'Initial Inventory': '20160.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_45.45', 'Revenue': '45.45', 'Demand': '4536', 'Initial Inventory': '30000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_60.6', 'Revenue': '60.6', 'Demand': '5601', 'Initial Inventory': '38360.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_75.75', 'Revenue': '75.75', 'Demand': '7567', 'Initial Inventory': '51450.0'}}]
import gurobipy as gp
from gurobipy import GRB
books_records = [rec for rec in LEGACY_RECORDS if rec['values']['Product_Name'].startswith('Books_')]
books = []
revenue = {}
demand = {}
inventory = {}
for rec in books_records:
    vals = rec['values']
    pname = vals['Product_Name']
    try:
        r = float(vals['Revenue'])
        d = int(float(vals['Demand']))
        s = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Invalid data for product {pname}: {e}')
    books.append(pname)
    revenue[pname] = r
    demand[pname] = d
    inventory[pname] = s
m = gp.Model('Books_Revenue_Max')
x_vars = m.addVars(books, lb=0, ub={p: min(demand[p], inventory[p]) for p in books}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in books)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')