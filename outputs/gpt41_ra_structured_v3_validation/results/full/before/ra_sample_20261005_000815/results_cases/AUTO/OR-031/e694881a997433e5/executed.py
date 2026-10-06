__lean_models_v2 = []

def __lean_capture_v2(value):
    __lean_models_v2.append(value)
    return value
CSVQA_DATA = {'ignored_file_indices': [], 'query': 'A retail enterprise sells multiple dairy products. The company aims to maximize total revenue by optimizing order fulfillment under initial inventory limits provided in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each product i that will be fulfilled.', 'relationships': [], 'route': 'NRM', 'tables': [{'columns': ['Full_Product_Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_index': 0, 'file_name': 'DairyGoodsSalesDataset.csv', 'filters': {}, 'original_rows': 40, 'records': [{'source_row': 0, 'values': {'Demand': '34102', 'Full_Product_Name': 'Butter_Amul', 'Initial Inventory': '29862', 'Revenue': '96.86'}}, {'source_row': 1, 'values': {'Demand': '36579', 'Full_Product_Name': 'Butter_Mother Dairy', 'Initial Inventory': '29898', 'Revenue': '48.01'}}, {'source_row': 2, 'values': {'Demand': '36086', 'Full_Product_Name': 'Butter_Parag Milk Foods', 'Initial Inventory': '25208', 'Revenue': '8.83'}}, {'source_row': 3, 'values': {'Demand': '41254', 'Full_Product_Name': 'Butter_Warana', 'Initial Inventory': '30816', 'Revenue': '92.96'}}, {'source_row': 4, 'values': {'Demand': '29876', 'Full_Product_Name': 'Buttermilk_Amul', 'Initial Inventory': '19925', 'Revenue': '40.75'}}, {'source_row': 5, 'values': {'Demand': '41229', 'Full_Product_Name': 'Buttermilk_Mother Dairy', 'Initial Inventory': '26482', 'Revenue': '83.07'}}, {'source_row': 6, 'values': {'Demand': '35354', 'Full_Product_Name': 'Buttermilk_Raj', 'Initial Inventory': '30865', 'Revenue': '15.64'}}, {'source_row': 7, 'values': {'Demand': '29649', 'Full_Product_Name': 'Buttermilk_Sudha', 'Initial Inventory': '33517', 'Revenue': '56.57'}}, {'source_row': 8, 'values': {'Demand': '38558', 'Full_Product_Name': 'Cheese_Amul', 'Initial Inventory': '30929', 'Revenue': '100.74'}}, {'source_row': 9, 'values': {'Demand': '28603', 'Full_Product_Name': 'Cheese_Britannia Industries', 'Initial Inventory': '21405', 'Revenue': '28.92'}}, {'source_row': 10, 'values': {'Demand': '35962', 'Full_Product_Name': 'Cheese_Dynamix Dairies', 'Initial Inventory': '25953', 'Revenue': '32.66'}}, {'source_row': 11, 'values': {'Demand': '36961', 'Full_Product_Name': 'Cheese_Passion Cheese', 'Initial Inventory': '23825', 'Revenue': '58.09'}}, {'source_row': 12, 'values': {'Demand': '39436', 'Full_Product_Name': 'Curd_Amul', 'Initial Inventory': '31687', 'Revenue': '30.27'}}, {'source_row': 13, 'values': {'Demand': '43522', 'Full_Product_Name': 'Curd_Mother Dairy', 'Initial Inventory': '33377', 'Revenue': '84.57'}}, {'source_row': 14, 'values': {'Demand': '38128', 'Full_Product_Name': 'Curd_Raj', 'Initial Inventory': '34914', 'Revenue': '84.75'}}, {'source_row': 15, 'values': {'Demand': '42341', 'Full_Product_Name': 'Curd_Sudha', 'Initial Inventory': '33547', 'Revenue': '76.37'}}, {'source_row': 16, 'values': {'Demand': '30345', 'Full_Product_Name': 'Ghee_Amul', 'Initial Inventory': '23120', 'Revenue': '41.49'}}, {'source_row': 17, 'values': {'Demand': '35420', 'Full_Product_Name': 'Ghee_Mother Dairy', 'Initial Inventory': '24667', 'Revenue': '52.79'}}, {'source_row': 18, 'values': {'Demand': '34100', 'Full_Product_Name': 'Ghee_Raj', 'Initial Inventory': '25395', 'Revenue': '48.13'}}, {'source_row': 19, 'values': {'Demand': '33007', 'Full_Product_Name': 'Ghee_Sudha', 'Initial Inventory': '24676', 'Revenue': '95.09'}}, {'source_row': 20, 'values': {'Demand': '37894', 'Full_Product_Name': 'Ice Cream_Amul', 'Initial Inventory': '26707', 'Revenue': '54.41'}}, {'source_row': 21, 'values': {'Demand': '29840', 'Full_Product_Name': 'Ice Cream_Dodla Dairy', 'Initial Inventory': '26722', 'Revenue': '82.24'}}, {'source_row': 22, 'values': {'Demand': '38762', 'Full_Product_Name': 'Ice Cream_Mother Dairy', 'Initial Inventory': '25809', 'Revenue': '94.32'}}, {'source_row': 23, 'values': {'Demand': '34674', 'Full_Product_Name': 'Ice Cream_Palle2patnam', 'Initial Inventory': '24391', 'Revenue': '83.73'}}, {'source_row': 24, 'values': {'Demand': '42972', 'Full_Product_Name': 'Lassi_Amul', 'Initial Inventory': '30728', 'Revenue': '74.45'}}, {'source_row': 25, 'values': {'Demand': '33894', 'Full_Product_Name': 'Lassi_Mother Dairy', 'Initial Inventory': '28628', 'Revenue': '49.4'}}, {'source_row': 26, 'values': {'Demand': '45762', 'Full_Product_Name': 'Lassi_Raj', 'Initial Inventory': '30568', 'Revenue': '93.93'}}, {'source_row': 27, 'values': {'Demand': '29503', 'Full_Product_Name': 'Lassi_Sudha', 'Initial Inventory': '23461', 'Revenue': '88.05'}}, {'source_row': 28, 'values': {'Demand': '34761', 'Full_Product_Name': 'Milk_Amul', 'Initial Inventory': '21398', 'Revenue': '39.24'}}, {'source_row': 29, 'values': {'Demand': '40548', 'Full_Product_Name': 'Milk_Mother Dairy', 'Initial Inventory': '33619', 'Revenue': '8.69'}}, {'source_row': 30, 'values': {'Demand': '43012', 'Full_Product_Name': 'Milk_Raj', 'Initial Inventory': '26355', 'Revenue': '65.53'}}, {'source_row': 31, 'values': {'Demand': '29180', 'Full_Product_Name': 'Milk_Sudha', 'Initial Inventory': '23815', 'Revenue': '42.34'}}, {'source_row': 32, 'values': {'Demand': '33498', 'Full_Product_Name': 'Paneer_Amul', 'Initial Inventory': '20787', 'Revenue': '81.76'}}, {'source_row': 33, 'values': {'Demand': '34848', 'Full_Product_Name': 'Paneer_Mother Dairy', 'Initial Inventory': '29342', 'Revenue': '29.09'}}, {'source_row': 34, 'values': {'Demand': '40347', 'Full_Product_Name': 'Paneer_Raj', 'Initial Inventory': '23556', 'Revenue': '87.3'}}, {'source_row': 35, 'values': {'Demand': '37188', 'Full_Product_Name': 'Paneer_Sudha', 'Initial Inventory': '28753', 'Revenue': '66.7'}}, {'source_row': 36, 'values': {'Demand': '34347', 'Full_Product_Name': 'Yogurt_Amul', 'Initial Inventory': '24404', 'Revenue': '89.32'}}, {'source_row': 37, 'values': {'Demand': '37181', 'Full_Product_Name': 'Yogurt_Dodla Dairy', 'Initial Inventory': '26829', 'Revenue': '33.81'}}, {'source_row': 38, 'values': {'Demand': '36644', 'Full_Product_Name': 'Yogurt_Mother Dairy', 'Initial Inventory': '25562', 'Revenue': '25.29'}}, {'source_row': 39, 'values': {'Demand': '34303', 'Full_Product_Name': 'Yogurt_Palle2patnam', 'Initial Inventory': '28695', 'Revenue': '84.9'}}], 'returned_rows': 40, 'role': 'dairy product revenue, demand, and inventory parameters', 'table_id': 'file_0_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
records = table['records']
products = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    prod = vals['Full_Product_Name']
    try:
        rev = float(vals['Revenue'])
        dem = int(vals['Demand'])
        inv = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Invalid data for product '{prod}': {e}")
    products.append(prod)
    revenue[prod] = rev
    demand[prod] = dem
    inventory[prod] = inv
if not set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()) == set(products):
    raise ValueError('Mismatch in product keys among revenue, demand, and inventory.')
m = __lean_capture_v2(gp.Model('DairyOrderFulfillment'))
x = m.addVars(products, lb=0, ub=[min(demand[i], inventory[i]) for i in products], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')