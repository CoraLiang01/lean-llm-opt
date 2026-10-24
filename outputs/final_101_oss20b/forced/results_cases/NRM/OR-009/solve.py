CSVQA_DATA = {'query': '“BrewCo,” a beverage manufacturer, operates multiple production facilities that distribute drinks to various retail locations. The daily demand for each retail outlet is specified in “customer_demand.csv,” while the production capacity of each plant is outlined in “supply_capacity.csv.” The transportation cost per unit of beverages from each plant to each outlet is recorded in “transportation_costs.csv.” The goal is to determine the optimal quantity of beverages to be shipped from each production plant to each retail outlet, ensuring all outlet demands are met without surpassing any plant’s production capacity, while minimizing the total transportation cost.', 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'customer_demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '94'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '39'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '65'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '435'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'supply_capacity'], 'file_name': 'supply_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '2531'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '20'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '210'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '241'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '543.756480860856', 'C2': '23.685276141764653', 'C3': '23.676386730773032', 'C4': '447.75143678673766', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '883.9151090405642', 'C2': '0.04977684765576961', 'C3': '0.0350986687216299', 'C4': '44.45588531711622', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '537.3456896658107', 'C2': '23.769274659075112', 'C3': '498.95659249465467', 'C4': '440.60737890439776', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '1791.493192397229', 'C2': '68.21633865655126', 'C3': '1432.4837339656747', 'C4': '1527.7635425462734', 'Unnamed: 0': 'S4'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():

    def get_table_by_id(table_id):
        for t in CSVQA_DATA['tables']:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    demand_table = get_table_by_id('file_0_view_0')
    d = {}
    for rec in demand_table['records']:
        values = rec['values']
        customer = values['customer']
        try:
            demand_val = float(values['demand'])
        except (ValueError, TypeError):
            raise ValueError(f'Invalid demand value for customer {customer}')
        d[customer] = demand_val
    supply_table = get_table_by_id('file_1_view_0')
    s = {}
    for rec in supply_table['records']:
        values = rec['values']
        plant = values['Unnamed: 0']
        try:
            supply_val = float(values['supply_capacity'])
        except (ValueError, TypeError):
            raise ValueError(f'Invalid supply capacity for plant {plant}')
        s[plant] = supply_val
    cost_table = get_table_by_id('file_2_view_0')
    c_cost = {}
    cost_columns = [col for col in cost_table['columns'] if col != 'Unnamed: 0']
    for rec in cost_table['records']:
        values = rec['values']
        plant = values['Unnamed: 0']
        for customer in cost_columns:
            try:
                cost_val = float(values[customer])
            except (ValueError, TypeError):
                raise ValueError(f'Invalid cost value for plant {plant}, customer {customer}')
            c_cost[plant, customer] = cost_val
    plants = list(s.keys())
    customers = list(d.keys())
    cost_plants = {rec['values']['Unnamed: 0'] for rec in cost_table['records']}
    if set(plants) != cost_plants:
        missing = set(plants) - cost_plants
        if missing:
            raise ValueError(f'Plants missing in cost matrix: {missing}')
        extra = cost_plants - set(plants)
        if extra:
            raise ValueError(f'Extra plants in cost matrix not in supply data: {extra}')
    if set(customers) != set(cost_columns):
        missing = set(customers) - set(cost_columns)
        if missing:
            raise ValueError(f'Customers missing in cost matrix columns: {missing}')
        extra = set(cost_columns) - set(customers)
        if extra:
            raise ValueError(f'Extra customers in cost matrix columns not in demand data: {extra}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(p, c) for p in plants for c in customers], lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((c_cost[p, c] * x[p, c] for p in plants for c in customers)), GRB.MINIMIZE)
    for p in plants:
        model.addConstr(gurobipy.quicksum((x[p, c] for c in customers)) <= s[p], name=f'supply_{p}')
    for c in customers:
        model.addConstr(gurobipy.quicksum((x[p, c] for p in plants)) == d[c], name=f'demand_{c}')
    model.optimize()
    return model
m = solve_problem()
