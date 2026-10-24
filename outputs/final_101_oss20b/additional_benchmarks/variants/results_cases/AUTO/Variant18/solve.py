CSVQA_DATA = {'query': 'A roll-cutting line has a fixed menu of feasible cutting patterns. Item demands are listed in item_demand.csv, and the number of each item produced by every pattern is listed in cutting_patterns.csv. A pattern may be used any nonnegative integer number of times, and producing extra pieces is allowed.\n\nFormulate a minimum-roll cutting-stock model. For each cutting pattern p, define y_p as the nonnegative integer number of stock rolls cut with pattern p. The objective is to minimize the total number of rolls used. The model should include demand satisfaction constraints for every item, nonnegativity constraints, and integer restrictions for all pattern-use variables.', 'relationships': [], 'tables': [{'columns': ['Item', 'Demand'], 'file_name': 'item_demand.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '18', 'Item': 'A'}}, {'data_row_number': 1, 'values': {'Demand': '14', 'Item': 'B'}}, {'data_row_number': 2, 'values': {'Demand': '12', 'Item': 'C'}}, {'data_row_number': 3, 'values': {'Demand': '10', 'Item': 'D'}}, {'data_row_number': 4, 'values': {'Demand': '8', 'Item': 'E'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Pattern', 'A', 'B', 'C', 'D', 'E'], 'file_name': 'cutting_patterns.csv', 'records': [{'data_row_number': 0, 'values': {'A': '3', 'B': '0', 'C': '0', 'D': '0', 'E': '0', 'Pattern': 'P1'}}, {'data_row_number': 1, 'values': {'A': '0', 'B': '2', 'C': '1', 'D': '0', 'E': '0', 'Pattern': 'P2'}}, {'data_row_number': 2, 'values': {'A': '0', 'B': '0', 'C': '2', 'D': '1', 'E': '0', 'Pattern': 'P3'}}, {'data_row_number': 3, 'values': {'A': '0', 'B': '0', 'C': '0', 'D': '2', 'E': '1', 'Pattern': 'P4'}}, {'data_row_number': 4, 'values': {'A': '1', 'B': '1', 'C': '0', 'D': '1', 'E': '0', 'Pattern': 'P5'}}, {'data_row_number': 5, 'values': {'A': '2', 'B': '0', 'C': '1', 'D': '0', 'E': '0', 'Pattern': 'P6'}}, {'data_row_number': 6, 'values': {'A': '0', 'B': '1', 'C': '1', 'D': '0', 'E': '1', 'Pattern': 'P7'}}, {'data_row_number': 7, 'values': {'A': '1', 'B': '0', 'C': '0', 'D': '1', 'E': '1', 'Pattern': 'P8'}}, {'data_row_number': 8, 'values': {'A': '1', 'B': '2', 'C': '0', 'D': '0', 'E': '0', 'Pattern': 'P9'}}, {'data_row_number': 9, 'values': {'A': '0', 'B': '0', 'C': '1', 'D': '1', 'E': '1', 'Pattern': 'P10'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    demand_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    pattern_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    items = [rec['values']['Item'] for rec in demand_table['records']]
    demand = {}
    for rec in demand_table['records']:
        item = rec['values']['Item']
        val = rec['values']['Demand']
        try:
            demand[item] = int(val)
        except ValueError:
            raise ValueError(f'Invalid demand value for item {item}: {val}')
    patterns = [rec['values']['Pattern'] for rec in pattern_table['records']]
    pattern_coeffs = {}
    for rec in pattern_table['records']:
        p = rec['values']['Pattern']
        coeffs = {}
        for item in items:
            if item not in rec['values']:
                raise ValueError(f'Missing coefficient for item {item} in pattern {p}')
            val = rec['values'][item]
            try:
                coeffs[item] = int(val)
            except ValueError:
                raise ValueError(f'Invalid coefficient for item {item} in pattern {p}: {val}')
        pattern_coeffs[p] = coeffs
    for p, coeffs in pattern_coeffs.items():
        if set(coeffs.keys()) != set(items):
            raise ValueError(f'Pattern {p} missing items')
    m = gurobipy.Model('CuttingStock')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(patterns, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((y[p] for p in patterns)), GRB.MINIMIZE)
    for i in items:
        expr = gurobipy.quicksum((pattern_coeffs[p][i] * y[p] for p in patterns))
        m.addConstr(expr >= demand[i], name=f'Demand_{i}')
    m.optimize()
    return m
m = solve_problem()
