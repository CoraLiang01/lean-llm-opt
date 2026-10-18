LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv","values":{"Product Name":"Baby Food_255.28","Revenue":"255.28","Demand":"765850","Initial Inventory":"5627060"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
record = LEGACY_RECORDS[0]
product_name = record['values']['Product Name']
revenue = float(record['values']['Revenue'])
demand = int(record['values']['Demand'])
initial_inventory = int(record['values']['Initial Inventory'])
m = gp.Model('Europe_Baby_Product_Fulfillment')
x_vars = m.addVars([product_name], lb=0, ub=min(demand, initial_inventory), vtype=GRB.INTEGER, name='')
m.setObjective(revenue * x_vars[product_name], GRB.MAXIMIZE)
m.addConstr(x_vars[product_name] <= demand, name='demand')
m.addConstr(x_vars[product_name] <= initial_inventory, name='inventory')
m.addConstr(x_vars[product_name] >= 0, name='nonneg')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')