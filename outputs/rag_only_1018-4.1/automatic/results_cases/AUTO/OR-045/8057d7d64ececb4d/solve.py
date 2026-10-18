LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv",\n    "values": {\n      "Capacity": "1035"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Spinach",\n      "Weight": "282",\n      "Value": "49"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Shiitake Mushrooms",\n      "Weight": "83",\n      "Value": "30"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Apples",\n      "Weight": "251",\n      "Value": "30"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Carrots",\n      "Weight": "257",\n      "Value": "18"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Basil",\n      "Weight": "88",\n      "Value": "54"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Potatoes",\n      "Weight": "52",\n      "Value": "27"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Green Beans",\n      "Weight": "198",\n      "Value": "91"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Blueberries",\n      "Weight": "203",\n      "Value": "88"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Oranges",\n      "Weight": "87",\n      "Value": "78"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Watermelons",\n      "Weight": "265",\n      "Value": "22"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv', 'values': {'Capacity': '1035'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '282', 'Value': '49'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '83', 'Value': '30'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Apples', 'Weight': '251', 'Value': '30'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '257', 'Value': '18'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Basil', 'Weight': '88', 'Value': '54'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '52', 'Value': '27'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '198', 'Value': '91'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '203', 'Value': '88'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '87', 'Value': '78'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '265', 'Value': '22'}}]
from gurobipy import Model, GRB, quicksum

def build_and_solve():
    products = []
    capacity = None
    for rec in LEGACY_RECORDS:
        if rec['source'].endswith('capacity.csv'):
            if capacity is not None:
                raise ValueError('Multiple capacities found')
            try:
                capacity = int(rec['values']['Capacity'])
            except Exception:
                raise ValueError('Invalid capacity value')
        elif rec['source'].endswith('products.csv'):
            v = rec['values']
            try:
                pname = v['ProductName']
                weight = int(v['Weight'])
                value = int(v['Value'])
            except Exception:
                raise ValueError('Invalid product record: %r' % v)
            products.append({'ProductName': pname, 'Weight': weight, 'Value': value})
    if capacity is None:
        raise ValueError('No capacity found')
    if not products:
        raise ValueError('No products found')
    prod_names = [p['ProductName'] for p in products]
    if len(set(prod_names)) != len(prod_names):
        raise ValueError('Duplicate product names')
    weights = {p['ProductName']: p['Weight'] for p in products}
    values = {p['ProductName']: p['Value'] for p in products}
    m = Model()
    m.Params.MIPGap = 0.0001
    order_vars = m.addVars(prod_names, vtype=GRB.INTEGER, lb=0, name='')
    for k in prod_names:
        if k not in weights or k not in values:
            raise ValueError('Missing data for product %r' % k)
    m.setObjective(quicksum((values[k] * order_vars[k] for k in prod_names)), GRB.MAXIMIZE)
    m.addConstr(quicksum((weights[k] * order_vars[k] for k in prod_names)) <= capacity, name='cap')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for k in prod_names:
            v = order_vars[k]
            print(v.VarName, v.X)
    else:
        print('Status', m.Status)
    return m
m = build_and_solve()