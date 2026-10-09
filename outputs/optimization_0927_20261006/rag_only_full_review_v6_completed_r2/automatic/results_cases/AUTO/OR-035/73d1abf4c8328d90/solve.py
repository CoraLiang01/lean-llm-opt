LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv",\n    "values": {\n      "Capacity": "180"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Baguette",\n      "Value": "888",\n      "Weight": "4"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Croissant",\n      "Value": "134",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Sourdough",\n      "Value": "129",\n      "Weight": "4"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Rye Bread",\n      "Value": "370",\n      "Weight": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Brioche",\n      "Value": "921",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Focaccia",\n      "Value": "765",\n      "Weight": "1"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Ciabatta",\n      "Value": "154",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Pita",\n      "Value": "837",\n      "Weight": "1"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Bagel",\n      "Value": "584",\n      "Weight": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "English Muffin",\n      "Value": "365",\n      "Weight": "3"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', 'values': {'Capacity': '180'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
from gurobipy import Model, GRB, quicksum
products = []
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('products.csv'):
        v = rec['values']
        products.append({'ProductName': v['ProductName'], 'Value': int(v['Value']), 'Weight': int(v['Weight'])})
    elif rec['source'].endswith('capacity.csv'):
        v = rec['values']
        if capacity is not None:
            raise ValueError('Multiple capacity records found')
        capacity = int(v['Capacity'])
if len(products) != 10:
    raise ValueError('Expected 10 products, got %d' % len(products))
if capacity is None:
    raise ValueError('No capacity record found')
product_keys = [p['ProductName'] for p in products]
values = {p['ProductName']: p['Value'] for p in products}
weights = {p['ProductName']: p['Weight'] for p in products}
for k in product_keys:
    if k not in values or k not in weights:
        raise ValueError(f'Missing data for product {k}')

def build_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(quicksum((values[k] * x_vars[k] for k in product_keys)), GRB.MAXIMIZE)
    m.addConstr(quicksum((weights[k] * x_vars[k] for k in product_keys)) <= capacity, name='storage')
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status', m.Status)