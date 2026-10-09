LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv","values":{"Product Name":"27in 4K Gaming Monitor","Revenue":"261.2933","Demand":"12474","Initial Inventory":"62440"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv","values":{"Product Name":"27in FHD Monitor","Revenue":"52.4965","Demand":"15057","Initial Inventory":"75500"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933', 'Demand': '12474', 'Initial Inventory': '62440'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '52.4965', 'Demand': '15057', 'Initial Inventory': '75500'}}]
import gurobipy as gp
from gurobipy import GRB
I = []
revenue = {}
demand = {}
initial_inventory = {}
capacity = {}
for record in LEGACY_RECORDS:
    values = record['values']
    product = values['Product Name']
    I.append(product)
    revenue[product] = float(values['Revenue'])
    demand[product] = int(values['Demand'])
    initial_inventory[product] = int(values['Initial Inventory'])
    capacity[product] = min(demand[product], initial_inventory[product])
for product in I:
    if product not in revenue or product not in demand or product not in initial_inventory or (product not in capacity):
        raise ValueError(f'Missing data for product: {product}')
m = gp.Model('27in_Product_Revenue_Maximization')
x_vars = m.addVars(I, lb=0, ub={i: capacity[i] for i in I}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')