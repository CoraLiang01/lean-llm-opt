LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv",\n    "values": {\n      "Capacity": "875"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Spinach",\n      "Weight": "230",\n      "Value": "64"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Shiitake Mushrooms",\n      "Weight": "637",\n      "Value": "75"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Apples",\n      "Weight": "773",\n      "Value": "68"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Carrots",\n      "Weight": "653",\n      "Value": "11"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Basil",\n      "Weight": "755",\n      "Value": "91"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Potatoes",\n      "Weight": "670",\n      "Value": "31"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Green Beans",\n      "Weight": "505",\n      "Value": "90"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Blueberries",\n      "Weight": "821",\n      "Value": "56"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Oranges",\n      "Weight": "83",\n      "Value": "10"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Watermelons",\n      "Weight": "249",\n      "Value": "24"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv', 'values': {'Capacity': '875'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'Value': '64'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'Value': '75'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Apples', 'Weight': '773', 'Value': '68'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'Value': '11'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Basil', 'Weight': '755', 'Value': '91'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'Value': '31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'Value': '90'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'Value': '56'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'Value': '10'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'Value': '24'}}]
from gurobipy import Model, GRB, quicksum
products = []
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('capacity.csv'):
        if capacity is not None:
            raise ValueError('Multiple capacities found in LEGACY_RECORDS')
        try:
            capacity = int(rec['values']['Capacity'])
        except Exception:
            raise ValueError('Invalid capacity value in LEGACY_RECORDS')
    elif rec['source'].endswith('products.csv'):
        try:
            pname = rec['values']['ProductName']
            weight = int(rec['values']['Weight'])
            value = int(rec['values']['Value'])
            products.append({'ProductName': pname, 'Weight': weight, 'Value': value})
        except Exception:
            raise ValueError('Invalid product record in LEGACY_RECORDS')
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS')
if len(products) == 0:
    raise ValueError('No products found in LEGACY_RECORDS')
if any(('Weight' not in p or 'Value' not in p for p in products)):
    raise ValueError('Missing product coefficients in LEGACY_RECORDS')
m = Model()
m.Params.MIPGap = 0.0001
x_vars = m.addVars(range(len(products)), vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((products[i]['Value'] * x_vars[i] for i in range(len(products)))), GRB.MAXIMIZE)
m.addConstr(quicksum((products[i]['Weight'] * x_vars[i] for i in range(len(products)))) <= capacity, name='cap')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for i in range(len(products)):
        print(x_vars[i].VarName, x_vars[i].X)
else:
    print(m.Status)