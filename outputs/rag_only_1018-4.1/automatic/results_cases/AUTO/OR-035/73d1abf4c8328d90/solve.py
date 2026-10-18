LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv",\n    "values": {\n      "Capacity": "180"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Baguette",\n      "Value": "888",\n      "Weight": "4"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Croissant",\n      "Value": "134",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Sourdough",\n      "Value": "129",\n      "Weight": "4"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Rye Bread",\n      "Value": "370",\n      "Weight": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Brioche",\n      "Value": "921",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Focaccia",\n      "Value": "765",\n      "Weight": "1"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Ciabatta",\n      "Value": "154",\n      "Weight": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Pita",\n      "Value": "837",\n      "Weight": "1"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "Bagel",\n      "Value": "584",\n      "Weight": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv",\n    "values": {\n      "ProductName": "English Muffin",\n      "Value": "365",\n      "Weight": "3"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', 'values': {'Capacity': '180'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
from gurobipy import Model, GRB, quicksum

def build_and_solve_bakery_model():
    capacity_records = [r for r in LEGACY_RECORDS if r['source'].endswith('capacity.csv')]
    if not capacity_records or 'Capacity' not in capacity_records[0]['values']:
        raise ValueError('Missing capacity data in LEGACY_RECORDS')
    try:
        total_capacity = int(capacity_records[0]['values']['Capacity'])
    except Exception:
        raise ValueError('Capacity value is not an integer')
    product_records = [r for r in LEGACY_RECORDS if r['source'].endswith('products.csv')]
    if not product_records:
        raise ValueError('No product records found in LEGACY_RECORDS')
    products = []
    profit = {}
    weight = {}
    for rec in product_records:
        vals = rec['values']
        if 'ProductName' not in vals or 'Value' not in vals or 'Weight' not in vals:
            raise ValueError('Missing fields in product record: %s' % vals)
        pname = vals['ProductName']
        try:
            profit[pname] = int(vals['Value'])
            weight[pname] = int(vals['Weight'])
        except Exception:
            raise ValueError('Non-integer Value or Weight for product %s' % pname)
        products.append(pname)
    if len(set(products)) != len(products):
        raise ValueError('Duplicate product names in LEGACY_RECORDS')
    m = Model()
    m.Params.MIPGap = 0.0001
    bread_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(quicksum((profit[p] * bread_vars[p] for p in products)), GRB.MAXIMIZE)
    m.addConstr(quicksum((weight[p] * bread_vars[p] for p in products)) <= total_capacity, name='storage')
    for p in products:
        if p not in profit or p not in weight:
            raise ValueError('Missing profit or weight for product %s' % p)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal:', m.ObjVal)
        for p in products:
            print(bread_vars[p].VarName, bread_vars[p].X)
    else:
        print('Solver status:', m.Status)
    return m
m = build_and_solve_bakery_model()