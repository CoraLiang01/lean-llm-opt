LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv",\n    "values": {\n      "Capacity": "1035"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Spinach",\n      "Weight": "282",\n      "Value": "49"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Shiitake Mushrooms",\n      "Weight": "83",\n      "Value": "30"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Apples",\n      "Weight": "251",\n      "Value": "30"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Carrots",\n      "Weight": "257",\n      "Value": "18"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Basil",\n      "Weight": "88",\n      "Value": "54"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Potatoes",\n      "Weight": "52",\n      "Value": "27"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Green Beans",\n      "Weight": "198",\n      "Value": "91"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Blueberries",\n      "Weight": "203",\n      "Value": "88"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Oranges",\n      "Weight": "87",\n      "Value": "78"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv",\n    "values": {\n      "ProductName": "Watermelons",\n      "Weight": "265",\n      "Value": "22"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv', 'values': {'Capacity': '1035'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '282', 'Value': '49'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '83', 'Value': '30'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Apples', 'Weight': '251', 'Value': '30'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '257', 'Value': '18'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Basil', 'Weight': '88', 'Value': '54'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '52', 'Value': '27'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '198', 'Value': '91'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '203', 'Value': '88'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '87', 'Value': '78'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '265', 'Value': '22'}}]
from gurobipy import Model, GRB, quicksum

def build_and_solve():
    global LEGACY_RECORDS
    capacity_records = [r for r in LEGACY_RECORDS if r['source'] and 'capacity.csv' in r['source'].lower()]
    if not capacity_records or 'Capacity' not in capacity_records[0]['values']:
        raise ValueError('Missing capacity record or field')
    try:
        total_capacity = int(capacity_records[0]['values']['Capacity'])
    except Exception:
        raise ValueError('Capacity value is not an integer')
    product_records = [r for r in LEGACY_RECORDS if r['source'] and 'products.csv' in r['source'].lower()]
    if not product_records:
        raise ValueError('No product records found')
    products = []
    weights = {}
    values = {}
    for rec in product_records:
        vals = rec['values']
        if not all((k in vals for k in ('ProductName', 'Weight', 'Value'))):
            raise ValueError('Missing product fields in record')
        pname = vals['ProductName']
        try:
            w = int(vals['Weight'])
            v = int(vals['Value'])
        except Exception:
            raise ValueError(f'Non-integer weight or value for product {pname}')
        products.append(pname)
        weights[pname] = w
        values[pname] = v
    if set(weights) != set(products) or set(values) != set(products):
        raise ValueError('Mismatch in product identifiers')
    m = Model()
    m.Params.MIPGap = 0.0001
    prod_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(quicksum((values[p] * prod_vars[p] for p in products)), GRB.MAXIMIZE)
    m.addConstr(quicksum((weights[p] * prod_vars[p] for p in products)) <= total_capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for p in products:
            print(f'{prod_vars[p].VarName} {prod_vars[p].X}')
    else:
        print(m.Status)
    return m
m = build_and_solve()