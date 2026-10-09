LEGACY_OBSERVATION = 'products.csv\ntwo_periods_ago_stock_status,previous_period_stock_status,item_name,previous_period_unit_value,item_value,two_periods_ago_unit_value,previous_period_replenishment_policy,previous_period_resource_requirement,resource_requirement\nBalanced,Stockout,Baguette,1044,888,853,Weekly,5,4\nOverstock,Overstock,Croissant,142,134,153,Daily,3,2\nOverstock,Balanced,Sourdough,141,129,105,Weekly,5,4\nOverstock,Overstock,Rye Bread,311,370,398,Weekly,4,3\nOverstock,Stockout,Brioche,753,921,951,On demand,3,2\nOverstock,Overstock,Focaccia,770,765,634,Weekly,2,1\nOverstock,Overstock,Ciabatta,129,154,127,On demand,3,2\nBalanced,Balanced,Pita,914,837,886,Daily,2,1\nStockout,Stockout,Bagel,668,584,605,Weekly,2,3\nStockout,Stockout,English Muffin,314,365,367,Daily,4,3\n\ncapacity.csv\nprevious_period_capacity,capacity_two_periods_ago,resource_capacity\n205,204,180'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'two_periods_ago_stock_status': 'Balanced', 'previous_period_stock_status': 'Stockout', 'item_name': 'Baguette', 'previous_period_unit_value': '1044', 'item_value': '888', 'two_periods_ago_unit_value': '853', 'previous_period_replenishment_policy': 'Weekly', 'previous_period_resource_requirement': '5', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'two_periods_ago_stock_status': 'Overstock', 'previous_period_stock_status': 'Overstock', 'item_name': 'Croissant', 'previous_period_unit_value': '142', 'item_value': '134', 'two_periods_ago_unit_value': '153', 'previous_period_replenishment_policy': 'Daily', 'previous_period_resource_requirement': '3', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'two_periods_ago_stock_status': 'Overstock', 'previous_period_stock_status': 'Balanced', 'item_name': 'Sourdough', 'previous_period_unit_value': '141', 'item_value': '129', 'two_periods_ago_unit_value': '105', 'previous_period_replenishment_policy': 'Weekly', 'previous_period_resource_requirement': '5', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'two_periods_ago_stock_status': 'Overstock', 'previous_period_stock_status': 'Overstock', 'item_name': 'Rye Bread', 'previous_period_unit_value': '311', 'item_value': '370', 'two_periods_ago_unit_value': '398', 'previous_period_replenishment_policy': 'Weekly', 'previous_period_resource_requirement': '4', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'two_periods_ago_stock_status': 'Overstock', 'previous_period_stock_status': 'Stockout', 'item_name': 'Brioche', 'previous_period_unit_value': '753', 'item_value': '921', 'two_periods_ago_unit_value': '951', 'previous_period_replenishment_policy': 'On demand', 'previous_period_resource_requirement': '3', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'two_periods_ago_stock_status': 'Overstock', 'previous_period_stock_status': 'Overstock', 'item_name': 'Focaccia', 'previous_period_unit_value': '770', 'item_value': '765', 'two_periods_ago_unit_value': '634', 'previous_period_replenishment_policy': 'Weekly', 'previous_period_resource_requirement': '2', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'two_periods_ago_stock_status': 'Overstock', 'previous_period_stock_status': 'Overstock', 'item_name': 'Ciabatta', 'previous_period_unit_value': '129', 'item_value': '154', 'two_periods_ago_unit_value': '127', 'previous_period_replenishment_policy': 'On demand', 'previous_period_resource_requirement': '3', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'two_periods_ago_stock_status': 'Balanced', 'previous_period_stock_status': 'Balanced', 'item_name': 'Pita', 'previous_period_unit_value': '914', 'item_value': '837', 'two_periods_ago_unit_value': '886', 'previous_period_replenishment_policy': 'Daily', 'previous_period_resource_requirement': '2', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'two_periods_ago_stock_status': 'Stockout', 'previous_period_stock_status': 'Stockout', 'item_name': 'Bagel', 'previous_period_unit_value': '668', 'item_value': '584', 'two_periods_ago_unit_value': '605', 'previous_period_replenishment_policy': 'Weekly', 'previous_period_resource_requirement': '2', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'two_periods_ago_stock_status': 'Stockout', 'previous_period_stock_status': 'Stockout', 'item_name': 'English Muffin', 'previous_period_unit_value': '314', 'item_value': '365', 'two_periods_ago_unit_value': '367', 'previous_period_replenishment_policy': 'Daily', 'previous_period_resource_requirement': '4', 'resource_requirement': '3'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '205', 'capacity_two_periods_ago': '204', 'resource_capacity': '180'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
profit = {}
resource_req = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        name = rec['values']['item_name']
        products.append(name)
        profit[name] = int(rec['values']['item_value'])
        resource_req[name] = int(rec['values']['resource_requirement'])
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        capacity = int(rec['values']['resource_capacity'])
        break
if capacity is None:
    raise ValueError('Missing resource capacity from LEGACY_RECORDS.')
if set(profit.keys()) != set(products) or set(resource_req.keys()) != set(products):
    raise ValueError('Profit or resource requirement data missing for some products.')
m = gp.Model('Bakery_Bread_Order')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((resource_req[i] * x[i] for i in products)) <= capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')