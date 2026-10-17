SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv', 'values': {'Capacity': '765'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Sedan', 'Value': '2524', 'Weight': '99'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'SUV', 'Value': '4614', 'Weight': '55'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Truck', 'Value': '8416', 'Weight': '75'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Convertible', 'Value': '5917', 'Weight': '94'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Minivan', 'Value': '9048', 'Weight': '80'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Coupe', 'Value': '1140', 'Weight': '82'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Hatchback', 'Value': '8962', 'Weight': '71'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Station Wagon', 'Value': '1888', 'Weight': '100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Electric Car', 'Value': '8487', 'Weight': '28'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Hybrid Car', 'Value': '4425', 'Weight': '93'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Luxury Sedan', 'Value': '4717', 'Weight': '84'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Sports Car', 'Value': '4210', 'Weight': '83'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Crossover', 'Value': '1226', 'Weight': '62'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Diesel Truck', 'Value': '7400', 'Weight': '90'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Compact SUV', 'Value': '4639', 'Weight': '99'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Luxury SUV', 'Value': '7712', 'Weight': '96'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Cargo Van', 'Value': '3299', 'Weight': '21'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Pickup Truck', 'Value': '9895', 'Weight': '39'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Roadster', 'Value': '4496', 'Weight': '99'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Muscle Car', 'Value': '4526', 'Weight': '81'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Off-road Vehicle', 'Value': '5688', 'Weight': '6'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Camper Van', 'Value': '3007', 'Weight': '58'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Compact Car', 'Value': '3623', 'Weight': '37'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Motorcycle', 'Value': '8474', 'Weight': '15'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', 'values': {'ProductName': 'Electric SUV', 'Value': '8372', 'Weight': '37'}}]
import gurobipy as gp
from gurobipy import GRB
records = SOURCE_RECORDS
capacity = None
for rec in records:
    if rec['source'].endswith('capacity.csv'):
        capacity = int(rec['values']['Capacity'])
if capacity is None:
    raise ValueError('Missing capacity record.')
products = []
value = {}
weight = {}
for rec in records:
    if rec['source'].endswith('products.csv'):
        pname = rec['values']['ProductName']
        products.append(pname)
        try:
            value[pname] = int(rec['values']['Value'])
            weight[pname] = int(rec['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid value/weight for {pname}: {e}')
if not products or len(products) != len(value) or len(products) != len(weight):
    raise ValueError('Mismatch or missing product/value/weight data.')
m = gp.Model('CarSales2_Inventory')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x_vars[p] for p in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')