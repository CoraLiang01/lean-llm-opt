LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv",\n    "values": {\n      "Capacity": "875"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Spinach",\n      "Weight": "230",\n      "Value": "64"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Shiitake Mushrooms",\n      "Weight": "637",\n      "Value": "75"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Apples",\n      "Weight": "773",\n      "Value": "68"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Carrots",\n      "Weight": "653",\n      "Value": "11"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Basil",\n      "Weight": "755",\n      "Value": "91"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Potatoes",\n      "Weight": "670",\n      "Value": "31"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Green Beans",\n      "Weight": "505",\n      "Value": "90"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Blueberries",\n      "Weight": "821",\n      "Value": "56"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Oranges",\n      "Weight": "83",\n      "Value": "10"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv",\n    "values": {\n      "ProductName": "Watermelons",\n      "Weight": "249",\n      "Value": "24"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv', 'values': {'Capacity': '875'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'Value': '64'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'Value': '75'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Apples', 'Weight': '773', 'Value': '68'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'Value': '11'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Basil', 'Weight': '755', 'Value': '91'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'Value': '31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'Value': '90'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'Value': '56'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'Value': '10'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'Value': '24'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
weights = {}
values = {}
capacity = None
for rec in LEGACY_RECORDS:
    src = rec['source']
    vals = rec['values']
    if src.endswith('products.csv'):
        pname = vals['ProductName']
        products.append(pname)
        weights[pname] = int(vals['Weight'])
        values[pname] = int(vals['Value'])
    elif src.endswith('capacity.csv'):
        if capacity is not None:
            raise ValueError('Multiple capacities found in LEGACY_RECORDS')
        capacity = int(vals['Capacity'])
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS')
if set(weights.keys()) != set(products) or set(values.keys()) != set(products):
    raise ValueError('Missing product data in LEGACY_RECORDS')
m = gp.Model('SupermarketStock')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in products)) <= capacity, name='stock_capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')