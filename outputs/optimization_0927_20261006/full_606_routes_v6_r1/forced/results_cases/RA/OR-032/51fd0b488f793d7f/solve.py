LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_15.15","Revenue":"15.15","Demand":"1980","Initial Inventory":"9920.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_30.3","Revenue":"30.3","Demand":"3024","Initial Inventory":"20160.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_45.45","Revenue":"45.45","Demand":"4536","Initial Inventory":"30000.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_60.6","Revenue":"60.6","Demand":"5601","Initial Inventory":"38360.0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv","values":{"Product_Name":"Books_75.75","Revenue":"75.75","Demand":"7567","Initial Inventory":"51450.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_15.15', 'Revenue': '15.15', 'Demand': '1980', 'Initial Inventory': '9920.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_30.3', 'Revenue': '30.3', 'Demand': '3024', 'Initial Inventory': '20160.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_45.45', 'Revenue': '45.45', 'Demand': '4536', 'Initial Inventory': '30000.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_60.6', 'Revenue': '60.6', 'Demand': '5601', 'Initial Inventory': '38360.0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv', 'values': {'Product_Name': 'Books_75.75', 'Revenue': '75.75', 'Demand': '7567', 'Initial Inventory': '51450.0'}}]
import gurobipy as gp
from gurobipy import GRB
books_records = [rec for rec in LEGACY_RECORDS if rec['values']['Product_Name'].startswith('Books_')]
products = []
revenue = {}
demand = {}
init_inventory = {}
for rec in books_records:
    name = rec['values']['Product_Name']
    products.append(name)
    try:
        revenue[name] = float(rec['values']['Revenue'])
        demand[name] = int(float(rec['values']['Demand']))
        init_inventory[name] = int(float(rec['values']['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Error parsing record for {name}: {e}')
upper_bound = {i: min(demand[i], init_inventory[i]) for i in products}
for i in products:
    if i not in revenue or i not in demand or i not in init_inventory:
        raise ValueError(f'Missing data for product {i}')
m = gp.Model('Books_Revenue_Maximization')
x_vars = m.addVars(products, lb=0, ub=[upper_bound[i] for i in products], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
for i in products:
    m.addConstr(x_vars[i] >= 0, name=f'nonneg_{i}')
    m.addConstr(x_vars[i] <= upper_bound[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')