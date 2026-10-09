LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv","values":{"Product Name":"Aalopuri","Revenue":"20","Demand":"1483","Initial Inventory":"10440.0"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv', 'values': {'Product Name': 'Aalopuri', 'Revenue': '20', 'Demand': '1483', 'Initial Inventory': '10440.0'}}]
import gurobipy as gp
from gurobipy import GRB
product = 'Aalopuri'
revenue = 20
demand = 1483
initial_inventory = 10440.0
m = gp.Model('Aalopuri_Revenue_Maximization')
x_vars = m.addVar(lb=0, ub=min(initial_inventory, demand), vtype=GRB.INTEGER, name='x')
m.setObjective(revenue * x_vars, GRB.MAXIMIZE)
m.addConstr(x_vars <= initial_inventory, name='inv')
m.addConstr(x_vars <= demand, name='dem')
m.addConstr(x_vars >= 0, name='nonneg')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')