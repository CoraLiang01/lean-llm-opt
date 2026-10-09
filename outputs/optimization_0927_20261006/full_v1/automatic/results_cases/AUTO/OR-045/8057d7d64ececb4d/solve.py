LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv","values":{"Capacity":"1035"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv","values":{"ProductName":"Spinach","Weight":"282","Value":"49"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv","values":{"ProductName":"Shiitake Mushrooms","Weight":"83","Value":"30"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv","values":{"ProductName":"Apples","Weight":"251","Value":"30"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv","values":{"ProductName":"Carrots","Weight":"257","Value":"18"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv","values":{"ProductName":"Basil","Weight":"88","Value":"54"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv","values":{"ProductName":"Potatoes","Weight":"52","Value":"27"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv","values":{"ProductName":"Green Beans","Weight":"198","Value":"91"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv","values":{"ProductName":"Blueberries","Weight":"203","Value":"88"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv","values":{"ProductName":"Oranges","Weight":"87","Value":"78"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv","values":{"ProductName":"Watermelons","Weight":"265","Value":"22"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv', 'values': {'Capacity': '1035'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '282', 'Value': '49'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '83', 'Value': '30'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Apples', 'Weight': '251', 'Value': '30'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '257', 'Value': '18'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Basil', 'Weight': '88', 'Value': '54'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '52', 'Value': '27'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '198', 'Value': '91'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '203', 'Value': '88'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '87', 'Value': '78'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '265', 'Value': '22'}}]
import gurobipy as gp
from gurobipy import GRB
capacity = None
products = []
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('capacity.csv') and rec['values']:
        if 'Capacity' in rec['values']:
            capacity = int(rec['values']['Capacity'])
    elif rec['source'].endswith('products.csv') and rec['values']:
        pname = rec['values']['ProductName']
        weight = int(rec['values']['Weight'])
        value = int(rec['values']['Value'])
        products.append({'ProductName': pname, 'Weight': weight, 'Value': value})
if capacity is None:
    raise ValueError('Missing capacity value in LEGACY_RECORDS')
if len(products) == 0:
    raise ValueError('No products found in LEGACY_RECORDS')
product_names = [p['ProductName'] for p in products]
weights = {p['ProductName']: p['Weight'] for p in products}
values = {p['ProductName']: p['Value'] for p in products}
m = gp.Model('Supermarket_Produce_Order')
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in product_names)) <= capacity, name='total_weight')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')