LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv","values":{"Capacity":"180"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Baguette","Value":"888","Weight":"4"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Croissant","Value":"134","Weight":"2"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Sourdough","Value":"129","Weight":"4"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Rye Bread","Value":"370","Weight":"3"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Brioche","Value":"921","Weight":"2"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Focaccia","Value":"765","Weight":"1"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Ciabatta","Value":"154","Weight":"2"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Pita","Value":"837","Weight":"1"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"Bagel","Value":"584","Weight":"3"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv","values":{"ProductName":"English Muffin","Value":"365","Weight":"3"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', 'values': {'Capacity': '180'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity_records = [r for r in records if r['source'].endswith('capacity.csv')]
if not capacity_records or 'Capacity' not in capacity_records[0]['values']:
    raise ValueError('Missing storage capacity in LEGACY_RECORDS')
try:
    storage_capacity = int(capacity_records[0]['values']['Capacity'])
except Exception:
    raise ValueError('Invalid storage capacity value in LEGACY_RECORDS')
product_records = [r for r in records if r['source'].endswith('products.csv')]
if not product_records:
    raise ValueError('No product records found in LEGACY_RECORDS')
products = []
values = {}
weights = {}
for rec in product_records:
    vals = rec['values']
    if 'ProductName' not in vals or 'Value' not in vals or 'Weight' not in vals:
        raise ValueError('Missing product fields in LEGACY_RECORDS')
    pname = vals['ProductName']
    try:
        v = int(vals['Value'])
        w = int(vals['Weight'])
    except Exception:
        raise ValueError(f'Invalid Value or Weight for product {pname}')
    products.append(pname)
    values[pname] = v
    weights[pname] = w
if set(values.keys()) != set(products) or set(weights.keys()) != set(products):
    raise ValueError('Mismatch in product identifiers for values/weights')
m = gp.Model('BakeryOrderOpt')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in products)) <= storage_capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')