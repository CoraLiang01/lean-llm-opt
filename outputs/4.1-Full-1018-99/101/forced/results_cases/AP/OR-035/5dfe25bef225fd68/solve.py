LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv",\n    "values": {\n      "Capacity": "180"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Baguette",\n      "Value": "888",\n      "Weight": "4"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Croissant",\n      "Value": "134",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Sourdough",\n      "Value": "129",\n      "Weight": "4"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Rye Bread",\n      "Value": "370",\n      "Weight": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Brioche",\n      "Value": "921",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Focaccia",\n      "Value": "765",\n      "Weight": "1"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Ciabatta",\n      "Value": "154",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Pita",\n      "Value": "837",\n      "Weight": "1"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Bagel",\n      "Value": "584",\n      "Weight": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "English Muffin",\n      "Value": "365",\n      "Weight": "3"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', 'values': {'Capacity': '180'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity = None
for rec in records:
    if rec['source'].endswith('capacity.csv'):
        val = rec['values'].get('Capacity')
        if val is not None:
            capacity = int(val)
if capacity is None:
    raise ValueError('Missing capacity in LEGACY_RECORDS')
products = []
for rec in records:
    if rec['source'].endswith('products.csv'):
        v = rec['values']
        pname = v.get('ProductName')
        value = v.get('Value')
        weight = v.get('Weight')
        if pname is None or value is None or weight is None:
            raise ValueError(f'Missing product fields in record: {v}')
        products.append({'ProductName': pname, 'Value': int(value), 'Weight': int(weight)})
for p in products:
    if not all((k in p for k in ('ProductName', 'Value', 'Weight'))):
        raise ValueError(f'Product missing fields: {p}')
product_names = [p['ProductName'] for p in products]
values = {p['ProductName']: p['Value'] for p in products}
weights = {p['ProductName']: p['Weight'] for p in products}
m = gp.Model('Bakery_Stocking')
x_vars = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in product_names)) <= capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')