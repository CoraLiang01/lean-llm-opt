SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv', 'values': {'Capacity': '586'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', 'values': {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
benefit = {}
requirement = {}
for rec in SOURCE_RECORDS:
    if rec['source'].endswith('products.csv'):
        name = rec['values']['ProductName']
        products.append(name)
        benefit[name] = int(rec['values']['Value'])
        requirement[name] = int(rec['values']['Weight'])
    elif rec['source'].endswith('capacity.csv'):
        capacity = int(rec['values']['Capacity'])
if len(products) == 0 or len(benefit) != len(products) or len(requirement) != len(products):
    raise ValueError('Missing or incomplete product data.')
m = gp.Model('NYC_Property_Development')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((benefit[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((requirement[p] * x_vars[p] for p in products)) <= capacity, name='dev_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')