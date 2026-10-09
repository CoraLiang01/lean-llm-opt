LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv",\n    "values": {\n      "Capacity": "1035"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Spinach",\n      "Weight": "282",\n      "Value": "49"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Shiitake Mushrooms",\n      "Weight": "83",\n      "Value": "30"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Apples",\n      "Weight": "251",\n      "Value": "30"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Carrots",\n      "Weight": "257",\n      "Value": "18"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Basil",\n      "Weight": "88",\n      "Value": "54"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Potatoes",\n      "Weight": "52",\n      "Value": "27"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Green Beans",\n      "Weight": "198",\n      "Value": "91"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Blueberries",\n      "Weight": "203",\n      "Value": "88"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Oranges",\n      "Weight": "87",\n      "Value": "78"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Watermelons",\n      "Weight": "265",\n      "Value": "22"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv', 'values': {'Capacity': '1035'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '282', 'Value': '49'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '83', 'Value': '30'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Apples', 'Weight': '251', 'Value': '30'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '257', 'Value': '18'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Basil', 'Weight': '88', 'Value': '54'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '52', 'Value': '27'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '198', 'Value': '91'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '203', 'Value': '88'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '87', 'Value': '78'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '265', 'Value': '22'}}]
import gurobipy as gp
from gurobipy import GRB
capacity = None
products = []
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('capacity.csv') and rec['values']:
        if 'Capacity' in rec['values']:
            if capacity is not None:
                raise ValueError('Multiple capacities found in LEGACY_RECORDS')
            capacity = int(rec['values']['Capacity'])
    elif rec['source'].endswith('products.csv') and rec['values']:
        if all((k in rec['values'] for k in ('ProductName', 'Weight', 'Value'))):
            products.append({'ProductName': rec['values']['ProductName'], 'Weight': int(rec['values']['Weight']), 'Value': int(rec['values']['Value'])})
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS')
if not products:
    raise ValueError('No products found in LEGACY_RECORDS')
product_names = [p['ProductName'] for p in products]
weights = {p['ProductName']: p['Weight'] for p in products}
values = {p['ProductName']: p['Value'] for p in products}
m = gp.Model('Supermarket_Produce_Order')
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[name] * x_vars[name] for name in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[name] * x_vars[name] for name in product_names)) <= capacity, name='total_weight')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')