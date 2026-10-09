LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv","values":{"Product Name":"27in 4K Gaming Monitor","Revenue":"261.2933","Demand":"12474","Initial Inventory":"62440"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv","values":{"Product Name":"27in FHD Monitor","Revenue":"52.4965","Demand":"15057","Initial Inventory":"75500"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933', 'Demand': '12474', 'Initial Inventory': '62440'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '52.4965', 'Demand': '15057', 'Initial Inventory': '75500'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product Name']
    products.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        demand[pname] = int(vals['Demand'])
        inventory[pname] = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Missing or invalid data for product '{pname}': {e}")
m = gp.Model('NRM13_27in')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x_vars[p] <= demand[p], name=f'demand_{p}')
    m.addConstr(x_vars[p] <= inventory[p], name=f'inventory_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')