LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv",\n    "values": {\n      "Capacity": "180"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Baguette",\n      "Value": "888",\n      "Weight": "4"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Croissant",\n      "Value": "134",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Sourdough",\n      "Value": "129",\n      "Weight": "4"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Rye Bread",\n      "Value": "370",\n      "Weight": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Brioche",\n      "Value": "921",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Focaccia",\n      "Value": "765",\n      "Weight": "1"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Ciabatta",\n      "Value": "154",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Pita",\n      "Value": "837",\n      "Weight": "1"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Bagel",\n      "Value": "584",\n      "Weight": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "English Muffin",\n      "Value": "365",\n      "Weight": "3"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', 'values': {'Capacity': '180'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
from gurobipy import Model, GRB, quicksum
products = []
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('capacity.csv'):
        if capacity is not None:
            raise ValueError('Multiple capacities found')
        capacity = int(rec['values']['Capacity'])
    elif rec['source'].endswith('products.csv'):
        pname = rec['values']['ProductName']
        val = rec['values']['Value']
        wt = rec['values']['Weight']
        if pname == '' or val == '' or wt == '':
            raise ValueError(f"Missing data in products.csv for {rec['values']}")
        products.append({'ProductName': pname, 'Value': int(val), 'Weight': int(wt)})
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS')
if len(products) == 0:
    raise ValueError('No products found in LEGACY_RECORDS')
product_keys = {'Baguette': 'x_Baguette', 'Croissant': 'x_Croissant', 'Sourdough': 'x_Sourdough', 'Rye Bread': 'x_RyeBread', 'Brioche': 'x_Brioche', 'Focaccia': 'x_Focaccia', 'Ciabatta': 'x_Ciabatta', 'Pita': 'x_Pita', 'Bagel': 'x_Bagel', 'English Muffin': 'x_EnglishMuffin'}
for pname in product_keys:
    if not any((p['ProductName'] == pname for p in products)):
        raise ValueError(f'Missing product data for {pname}')
profit = {}
weight = {}
for p in products:
    key = product_keys[p['ProductName']]
    profit[key] = p['Value']
    weight[key] = p['Weight']
m = Model()
m.setParam('MIPGap', 0.0001)
x_vars = m.addVars(product_keys.values(), vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((profit[k] * x_vars[k] for k in product_keys.values())), GRB.MAXIMIZE)
m.addConstr(quicksum((weight[k] * x_vars[k] for k in product_keys.values())) <= capacity, name='storage')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for k in product_keys.values():
        print(f'{x_vars[k].VarName}: {x_vars[k].X}')
else:
    print(f'Solver status: {m.Status}')