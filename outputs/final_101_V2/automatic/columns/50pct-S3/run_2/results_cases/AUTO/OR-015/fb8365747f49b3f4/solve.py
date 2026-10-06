LEGACY_OBSERVATION = 'capacity.csv\n\nprevious_period_capacity,resource_id,resource_capacity\n564,1,500\n679,2,700\n481,3,600\n684,4,800\n623,5,550\n1019,6,900\n671,7,650\n761,8,750\n951,9,820\n522,10,570\n\nproducts.csv\n\nprevious_period_stock_status,item_name,item_value,resource_requirement,previous_period_unit_value\nStockout,1,50,10,48\nStockout,2,70,20,56\nStockout,3,30,5,31\nStockout,4,60,15,53\nOverstock,5,80,25,96\nOverstock,6,90,30,91\nStockout,7,40,12,45\nStockout,8,100,35,119\nStockout,9,55,10,62\nBalanced,10,75,20,60\nStockout,11,65,18,67\nBalanced,12,95,28,84\nBalanced,13,45,8,38\nBalanced,14,85,22,68\nBalanced,15,70,25,74\nBalanced,16,110,40,119\nBalanced,17,50,14,42\nOverstock,18,60,16,48\nOverstock,19,120,50,117\nOverstock,20,100,30,93'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'previous_period_capacity': '564', 'resource_id': '1', 'resource_capacity': '500'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '679', 'resource_id': '2', 'resource_capacity': '700'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '481', 'resource_id': '3', 'resource_capacity': '600'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '684', 'resource_id': '4', 'resource_capacity': '800'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '623', 'resource_id': '5', 'resource_capacity': '550'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '1019', 'resource_id': '6', 'resource_capacity': '900'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '671', 'resource_id': '7', 'resource_capacity': '650'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '761', 'resource_id': '8', 'resource_capacity': '750'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '951', 'resource_id': '9', 'resource_capacity': '820'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '522', 'resource_id': '10', 'resource_capacity': '570'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': '1', 'item_value': '50', 'resource_requirement': '10', 'previous_period_unit_value': '48'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': '2', 'item_value': '70', 'resource_requirement': '20', 'previous_period_unit_value': '56'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': '3', 'item_value': '30', 'resource_requirement': '5', 'previous_period_unit_value': '31'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': '4', 'item_value': '60', 'resource_requirement': '15', 'previous_period_unit_value': '53'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': '5', 'item_value': '80', 'resource_requirement': '25', 'previous_period_unit_value': '96'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': '6', 'item_value': '90', 'resource_requirement': '30', 'previous_period_unit_value': '91'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': '7', 'item_value': '40', 'resource_requirement': '12', 'previous_period_unit_value': '45'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': '8', 'item_value': '100', 'resource_requirement': '35', 'previous_period_unit_value': '119'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': '9', 'item_value': '55', 'resource_requirement': '10', 'previous_period_unit_value': '62'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': '10', 'item_value': '75', 'resource_requirement': '20', 'previous_period_unit_value': '60'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': '11', 'item_value': '65', 'resource_requirement': '18', 'previous_period_unit_value': '67'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': '12', 'item_value': '95', 'resource_requirement': '28', 'previous_period_unit_value': '84'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': '13', 'item_value': '45', 'resource_requirement': '8', 'previous_period_unit_value': '38'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': '14', 'item_value': '85', 'resource_requirement': '22', 'previous_period_unit_value': '68'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': '15', 'item_value': '70', 'resource_requirement': '25', 'previous_period_unit_value': '74'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': '16', 'item_value': '110', 'resource_requirement': '40', 'previous_period_unit_value': '119'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': '17', 'item_value': '50', 'resource_requirement': '14', 'previous_period_unit_value': '42'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': '18', 'item_value': '60', 'resource_requirement': '16', 'previous_period_unit_value': '48'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': '19', 'item_value': '120', 'resource_requirement': '50', 'previous_period_unit_value': '117'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': '20', 'item_value': '100', 'resource_requirement': '30', 'previous_period_unit_value': '93'}}]
import gurobipy as gp
from gurobipy import GRB
capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
shelves = []
resource_capacity = {}
for rec in capacity_records:
    rid = str(rec['values']['resource_id'])
    shelves.append(rid)
    resource_capacity[rid] = int(rec['values']['resource_capacity'])
products = []
item_value = {}
resource_requirement = {}
for rec in product_records:
    pname = str(rec['values']['item_name'])
    products.append(pname)
    item_value[pname] = int(rec['values']['item_value'])
    resource_requirement[pname] = int(rec['values']['resource_requirement'])
if len(shelves) != 10:
    raise ValueError('Expected 10 shelves, got %d' % len(shelves))
if len(products) != 20:
    raise ValueError('Expected 20 products, got %d' % len(products))
if set(resource_capacity.keys()) != set(shelves):
    raise ValueError('Mismatch in shelf/resource_capacity keys')
if set(item_value.keys()) != set(products) or set(resource_requirement.keys()) != set(products):
    raise ValueError('Mismatch in product keys for value/requirement')
m = gp.Model('BigMart_Shelf_Allocation')
x = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((item_value[j] * x[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((resource_requirement[j] * x[i, j] for j in products)) <= resource_capacity[i] for i in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')