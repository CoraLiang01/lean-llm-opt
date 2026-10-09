LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv","values":{"Capacity":"180"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Baguette","Value":"888","Weight":"4"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Croissant","Value":"134","Weight":"2"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Sourdough","Value":"129","Weight":"4"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Rye Bread","Value":"370","Weight":"3"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Brioche","Value":"921","Weight":"2"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Focaccia","Value":"765","Weight":"1"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Ciabatta","Value":"154","Weight":"2"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Pita","Value":"837","Weight":"1"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Bagel","Value":"584","Weight":"3"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"English Muffin","Value":"365","Weight":"3"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', 'values': {'Capacity': '180'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity = None
for rec in records:
    if rec['source'].endswith('capacity.csv'):
        capacity = int(rec['values']['Capacity'])
if capacity is None:
    raise ValueError('Missing capacity data.')
product_order = ['Baguette', 'Croissant', 'Sourdough', 'Rye Bread', 'Brioche', 'Focaccia', 'Ciabatta', 'Pita', 'Bagel', 'English Muffin']
products = []
profit = {}
weight = {}
for pname in product_order:
    found = False
    for rec in records:
        if rec['source'].endswith('products.csv') and rec['values']['ProductName'] == pname:
            products.append(pname)
            profit[pname] = int(rec['values']['Value'])
            weight[pname] = int(rec['values']['Weight'])
            found = True
            break
    if not found:
        raise ValueError(f'Missing product data for {pname}')
m = gp.Model('Bakery_Stocking')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x_vars[p] for p in products)) <= capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')