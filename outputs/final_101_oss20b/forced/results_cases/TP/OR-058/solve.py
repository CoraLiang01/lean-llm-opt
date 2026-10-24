CSVQA_DATA = {'query': 'Several suppliers located in different regions can provide the necessary Adidas products to various stores. Each supplier incurs a fixed cost when starting operations, with the fixed cost data provided in “fixed_cost.csv.” The company needs to source a unit of each Adidas product for the stores from these suppliers. For each product, the transportation cost per unit from each supplier to each store is recorded in “transportation_costs.csv.” Additionally, each store has a specific demand for these products, which is provided in “demand.csv.” The objective is to determine which suppliers to activate so that the demand for all Adidas products across all stores is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that each store S_j sources from supplier F_i.', 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '216'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '216'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '216'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '144'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '144'}}, {'data_row_number': 5, 'values': {'customer': 'C6', 'demand': '144'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '0.08', 'C2': '52.33', 'C3': '73.56999999999999', 'C4': '1237.33', 'C5': '0.07000000000000001', 'C6': '112.16', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '46.02', 'C2': '175.23', 'C3': '2026.83', 'C4': '299.89', 'C5': '966.53', 'C6': '1590.42', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '1031.74', 'C2': '78.13', 'C3': '99.02', 'C4': '277.07', 'C5': '884.45', 'C6': '1800.86', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '868.75', 'C2': '94.2', 'C3': '1776.34', 'C4': '285.48', 'C5': '868.85', 'C6': '86.55', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '1577', 'C2': '760.15', 'C3': '2090.19', 'C4': '43.2', 'C5': '1577.12', 'C6': '1095.17', 'Unnamed: 0': 'S5'}}, {'data_row_number': 5, 'values': {'C1': '49.14', 'C2': '4.33', 'C3': '2079.57', 'C4': '277.04', 'C5': '1032.01', 'C6': '1543.49', 'Unnamed: 0': 'S6'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    table_demand = tables['file_0_view_0']
    demand = {}
    for rec in table_demand['records']:
        cust = rec['values']['customer']
        demand[cust] = float(rec['values']['demand'])
    table_fixed = tables['file_1_view_0']
    suppliers = [rec['values']['Unnamed: 0'] for rec in table_fixed['records']]
    fixed_costs = {}
    for rec in table_fixed['records']:
        sup = rec['values']['Unnamed: 0']
        fixed_costs[sup] = float(rec['values']['fixed_costs'])
    table_costs = tables['file_2_view_0']
    customers = [col for col in table_costs['columns'] if col.startswith('C')]
    costs = {}
    for rec in table_costs['records']:
        sup = rec['values']['Unnamed: 0']
        costs[sup] = {}
        for cust in customers:
            costs[sup][cust] = float(rec['values'][cust])
    for sup in suppliers:
        if sup not in fixed_costs:
            raise ValueError(f'Missing fixed cost for supplier {sup}')
        if sup not in costs:
            raise ValueError(f'Missing cost matrix for supplier {sup}')
        for cust in customers:
            if cust not in costs[sup]:
                raise ValueError(f'Missing cost for supplier {sup} to customer {cust}')
    for cust in customers:
        if cust not in demand:
            raise ValueError(f'Missing demand for customer {cust}')
    total_demand = sum(demand.values())
    M = total_demand
    m = gurobipy.Model('Supplier_Activation_TP')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in suppliers for j in customers], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((fixed_costs[i] * y[i] for i in suppliers)) + gurobipy.quicksum((costs[i][j] * x[i, j] for i in suppliers for j in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((x[i, j] <= M * y[i] for i in suppliers for j in customers), name='')
    m.optimize()
    return m
m = solve_problem()
