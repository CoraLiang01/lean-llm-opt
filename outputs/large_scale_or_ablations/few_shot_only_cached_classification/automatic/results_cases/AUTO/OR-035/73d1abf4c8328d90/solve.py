LEGACY_OBSERVATION = '{"ignored_file_indices": [], "query": "A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of bread, we have an expected profit, which can be found in \\"products.csv.\\" However, the shop has limited storage capacity, with details provided in \\"capacity.csv.\\".Therefore, we must decide which types of bread to order each day to maximize our total expected profit while staying within our storage limits. The decision variables x_i represents the number of units of bread type i to be ordered each day.The decision variables must be integers.", "relationships": [], "route": "RA", "tables": [{"columns": ["Capacity"], "file_index": 0, "file_name": "capacity.csv", "filters": {"conditions": [], "logic": "and"}, "original_rows": 1, "records": [{"source_row": 0, "values": {"Capacity": "180"}}], "returned_rows": 1, "role": "file_0", "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv", "table_id": "file_0_view_0"}, {"columns": ["ProductName", "Value", "Weight"], "file_index": 1, "file_name": "products.csv", "filters": {"conditions": [], "logic": "and"}, "original_rows": 10, "records": [{"source_row": 0, "values": {"ProductName": "Baguette", "Value": "888", "Weight": "4"}}, {"source_row": 1, "values": {"ProductName": "Croissant", "Value": "134", "Weight": "2"}}, {"source_row": 2, "values": {"ProductName": "Sourdough", "Value": "129", "Weight": "4"}}, {"source_row": 3, "values": {"ProductName": "Rye Bread", "Value": "370", "Weight": "3"}}, {"source_row": 4, "values": {"ProductName": "Brioche", "Value": "921", "Weight": "2"}}, {"source_row": 5, "values": {"ProductName": "Focaccia", "Value": "765", "Weight": "1"}}, {"source_row": 6, "values": {"ProductName": "Ciabatta", "Value": "154", "Weight": "2"}}, {"source_row": 7, "values": {"ProductName": "Pita", "Value": "837", "Weight": "1"}}, {"source_row": 8, "values": {"ProductName": "Bagel", "Value": "584", "Weight": "3"}}, {"source_row": 9, "values": {"ProductName": "English Muffin", "Value": "365", "Weight": "3"}}], "returned_rows": 10, "role": "file_1", "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv", "table_id": "file_1_view_0"}], "validation": {"status": "DIRECT_CSV_FULL_DATA"}}'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', 'values': {'Capacity': '180'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
profit = {}
weight = {}
capacity = None
for rec in records:
    src = rec['source']
    vals = rec['values']
    if src and src.endswith('products.csv'):
        pname = vals['ProductName']
        products.append(pname)
        profit[pname] = int(vals['Value'])
        weight[pname] = int(vals['Weight'])
    elif src and src.endswith('capacity.csv'):
        if capacity is not None:
            raise ValueError('Multiple capacities found')
        capacity = int(vals['Capacity'])
if capacity is None:
    raise ValueError('No capacity found')
if set(profit.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Missing profit or weight data for some products')
m = gp.Model('BakeryOrder')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in products)) <= capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')