CSVQA_DATA = {'query': 'A logistics company manages multiple warehouses responsible for distributing products to various retail stores. The daily demand for each store is provided in “customer_demand.csv,” while the daily supply capacity of each warehouse is detailed in “supply_capacity.csv.” The cost of transporting each unit of product from each warehouse to each store is recorded in “transportation_costs.csv.” The objective is to determine the optimal quantity of products to be shipped from each warehouse to each store, ensuring that all store demands are met without exceeding the supply capacity of any warehouse, while minimizing the total transportation cost.', 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'customer_demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '45'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '23'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '94'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '92'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '57'}}, {'data_row_number': 5, 'values': {'customer': 'C6', 'demand': '52'}}, {'data_row_number': 6, 'values': {'customer': 'C7', 'demand': '23'}}, {'data_row_number': 7, 'values': {'customer': 'C8', 'demand': '99'}}, {'data_row_number': 8, 'values': {'customer': 'C9', 'demand': '99'}}, {'data_row_number': 9, 'values': {'customer': 'C10', 'demand': '77'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'supply_capacity'], 'file_name': 'supply_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '127'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '236'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '168'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '115'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'supply_capacity': '280'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'S6', 'supply_capacity': '179'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'S7', 'supply_capacity': '135'}}, {'data_row_number': 7, 'values': {'Unnamed: 0': 'S8', 'supply_capacity': '263'}}, {'data_row_number': 8, 'values': {'Unnamed: 0': 'S9', 'supply_capacity': '283'}}, {'data_row_number': 9, 'values': {'Unnamed: 0': 'S10', 'supply_capacity': '476'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '2077.058672521021', 'C2': '0.0', 'C3': '54.33526480458508', 'C4': '0.0', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '2077.058672521021', 'C2': '0.0', 'C3': '1141.0405608962865', 'C4': '0.0', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '79.9210295982608', 'C2': '474.24509131006675', 'C3': '1477.0676289106607', 'C4': '22.583099586193658', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '1659.336929105112', 'C2': '57.20541468776147', 'C3': '186.1519048103841', 'C4': '1201.3137084429907', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '1297.2567040858307', 'C2': '77.76629131320436', 'C3': '24.26760227579214', 'C4': '1399.7932436376784', 'Unnamed: 0': 'S5'}}, {'data_row_number': 5, 'values': {'C1': '1998.9090658724567', 'C2': '985.3165435695341', 'C3': '2.8541686885814643', 'C4': '1149.53596749779', 'Unnamed: 0': 'S6'}}, {'data_row_number': 6, 'values': {'C1': '1780.3360050180179', 'C2': '0.0', 'C3': '1141.0405608962865', 'C4': '0.0', 'Unnamed: 0': 'S7'}}, {'data_row_number': 7, 'values': {'C1': '75.40935896233042', 'C2': '1338.1987290721909', 'C3': '21.391345987195333', 'C4': '74.34437383734394', 'Unnamed: 0': 'S8'}}, {'data_row_number': 8, 'values': {'C1': '98.90755583433433', 'C2': '0.0', 'C3': '978.0347664825314', 'C4': '0.0', 'Unnamed: 0': 'S9'}}, {'data_row_number': 9, 'values': {'C1': '2077.058672521021', 'C2': '0.0', 'C3': '54.33526480458508', 'C4': '0.0', 'Unnamed: 0': 'S10'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    demand_table = tables['file_0_view_0']
    demand = {}
    for rec in demand_table['records']:
        cust = rec['values']['customer']
        val = rec['values']['demand']
        try:
            demand[cust] = int(val)
        except ValueError:
            raise ValueError(f'Invalid demand value for customer {cust}: {val}')
    supply_table = tables['file_1_view_0']
    supply = {}
    for rec in supply_table['records']:
        wh = rec['values']['Unnamed: 0']
        val = rec['values']['supply_capacity']
        try:
            supply[wh] = int(val)
        except ValueError:
            raise ValueError(f'Invalid supply capacity for warehouse {wh}: {val}')
    cost_table = tables['file_2_view_0']
    cost = {}
    for rec in cost_table['records']:
        wh = rec['values']['Unnamed: 0']
        for cust_col in rec['values']:
            if cust_col.startswith('C'):
                cust = cust_col
                val = rec['values'][cust_col]
                if val == '' or val is None:
                    continue
                try:
                    cost[wh, cust] = float(val)
                except ValueError:
                    raise ValueError(f'Invalid cost value for warehouse {wh}, customer {cust}: {val}')
    missing_cost_customers = [c for c in demand if not any(((w, c) in cost for w in supply))]
    if missing_cost_customers:
        raise ValueError(f"Missing cost entries for customers: {', '.join(missing_cost_customers)}")
    missing_supply_warehouses = [w for w in cost if w not in supply]
    if missing_supply_warehouses:
        raise ValueError(f"Missing supply capacity for warehouses: {', '.join(missing_supply_warehouses)}")
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(cost.keys(), lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((cost[k] * x[k] for k in cost)), GRB.MINIMIZE)
    for cust in demand:
        m.addConstr(gurobipy.quicksum((x[wh, cust] for wh in supply if (wh, cust) in cost)) == demand[cust], name=f'demand_{cust}')
    for wh in supply:
        m.addConstr(gurobipy.quicksum((x[wh, cust] for cust in demand if (wh, cust) in cost)) <= supply[wh], name=f'supply_{wh}')
    m.optimize()
    return m
m = solve_problem()
