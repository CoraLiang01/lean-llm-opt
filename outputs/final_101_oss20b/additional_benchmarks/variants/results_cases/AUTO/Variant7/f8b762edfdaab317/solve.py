CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A distribution network ships goods from source plants to customers through intermediate cross-dock hubs. '
          'Source supplies and customer demands are listed in node_supply_demand.csv, hub throughput capacities are '
          'listed in hub_capacity.csv, and per-unit transportation costs for source-to-hub and hub-to-customer arcs '
          'are listed in arc_costs.csv. The total shipments out of each source may not exceed its available supply; '
          'unused source supply is allowed.\n'
          '\n'
          'Formulate a minimum-cost transshipment model. For each directed arc i-j, define f_ij as the nonnegative '
          'shipment flow on that arc. The objective is to minimize total transportation cost. The model should include '
          'source supply upper-bound constraints, customer demand constraints, flow-balance constraints at each hub, '
          'hub throughput-capacity constraints, and nonnegativity constraints for all arc-flow variables.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Node', 'NodeType', 'Amount'],
             'file_index': 0,
             'file_name': 'node_supply_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0, 'values': {'Amount': '120', 'Node': 'S1', 'NodeType': 'SourceSupply'}},
                         {'source_row': 1, 'values': {'Amount': '100', 'Node': 'S2', 'NodeType': 'SourceSupply'}},
                         {'source_row': 2, 'values': {'Amount': '90', 'Node': 'S3', 'NodeType': 'SourceSupply'}},
                         {'source_row': 3, 'values': {'Amount': '70', 'Node': 'C1', 'NodeType': 'CustomerDemand'}},
                         {'source_row': 4, 'values': {'Amount': '80', 'Node': 'C2', 'NodeType': 'CustomerDemand'}},
                         {'source_row': 5, 'values': {'Amount': '60', 'Node': 'C3', 'NodeType': 'CustomerDemand'}},
                         {'source_row': 6, 'values': {'Amount': '90', 'Node': 'C4', 'NodeType': 'CustomerDemand'}}],
             'returned_rows': 7,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Hub', 'ThroughputCapacity'],
             'file_index': 1,
             'file_name': 'hub_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Hub': 'H1', 'ThroughputCapacity': '170'}},
                         {'source_row': 1, 'values': {'Hub': 'H2', 'ThroughputCapacity': '160'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['From', 'To', 'Cost'],
             'file_index': 2,
             'file_name': 'arc_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 14,
             'records': [{'source_row': 0, 'values': {'Cost': '2', 'From': 'S1', 'To': 'H1'}},
                         {'source_row': 1, 'values': {'Cost': '6', 'From': 'S1', 'To': 'H2'}},
                         {'source_row': 2, 'values': {'Cost': '4', 'From': 'S2', 'To': 'H1'}},
                         {'source_row': 3, 'values': {'Cost': '3', 'From': 'S2', 'To': 'H2'}},
                         {'source_row': 4, 'values': {'Cost': '7', 'From': 'S3', 'To': 'H1'}},
                         {'source_row': 5, 'values': {'Cost': '2', 'From': 'S3', 'To': 'H2'}},
                         {'source_row': 6, 'values': {'Cost': '3', 'From': 'H1', 'To': 'C1'}},
                         {'source_row': 7, 'values': {'Cost': '4', 'From': 'H1', 'To': 'C2'}},
                         {'source_row': 8, 'values': {'Cost': '7', 'From': 'H1', 'To': 'C3'}},
                         {'source_row': 9, 'values': {'Cost': '8', 'From': 'H1', 'To': 'C4'}},
                         {'source_row': 10, 'values': {'Cost': '8', 'From': 'H2', 'To': 'C1'}},
                         {'source_row': 11, 'values': {'Cost': '6', 'From': 'H2', 'To': 'C2'}},
                         {'source_row': 12, 'values': {'Cost': '3', 'From': 'H2', 'To': 'C3'}},
                         {'source_row': 13, 'values': {'Cost': '4', 'From': 'H2', 'To': 'C4'}}],
             'returned_rows': 14,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    node_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    hub_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    arc_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    supply_s = {}
    demand_c = {}
    for rec in node_table['records']:
        node = rec['values']['Node']
        node_type = rec['values']['NodeType']
        amount = float(rec['values']['Amount'])
        if node_type == 'SourceSupply':
            supply_s[node] = amount
        elif node_type == 'CustomerDemand':
            demand_c[node] = amount
    S = set(supply_s.keys())
    C = set(demand_c.keys())
    cap_h = {}
    for rec in hub_table['records']:
        hub = rec['values']['Hub']
        cap = float(rec['values']['ThroughputCapacity'])
        cap_h[hub] = cap
    H = set(cap_h.keys())
    cost_ij = {}
    arcs = []
    for rec in arc_table['records']:
        frm = rec['values']['From']
        to = rec['values']['To']
        cost = float(rec['values']['Cost'])
        arcs.append((frm, to))
        cost_ij[frm, to] = cost
    for s in S:
        if s not in supply_s:
            raise ValueError(f'Missing supply for source {s}')
    for c in C:
        if c not in demand_c:
            raise ValueError(f'Missing demand for customer {c}')
    for h in H:
        if h not in cap_h:
            raise ValueError(f'Missing capacity for hub {h}')
    for arc in arcs:
        if arc not in cost_ij:
            raise ValueError(f'Missing cost for arc {arc[0]}->{arc[1]}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    f = m.addVars(arcs, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost_ij[arc] * f[arc] for arc in arcs)), GRB.MINIMIZE)
    for s in S:
        expr = gurobipy.quicksum((f[s, j] for s2, j in arcs if s2 == s and j in H))
        m.addConstr(expr <= supply_s[s], name=f'supply_{s}')
    for c in C:
        expr = gurobipy.quicksum((f[i, c] for i, c2 in arcs if c2 == c and i in H))
        m.addConstr(expr >= demand_c[c], name=f'demand_{c}')
    for h in H:
        inflow = gurobipy.quicksum((f[i, h] for i, h2 in arcs if h2 == h and i in S))
        outflow = gurobipy.quicksum((f[h, j] for h2, j in arcs if h2 == h and j in C))
        m.addConstr(inflow == outflow, name=f'balance_{h}')
    for h in H:
        inflow = gurobipy.quicksum((f[i, h] for i, h2 in arcs if h2 == h and i in S))
        m.addConstr(inflow <= cap_h[h], name=f'cap_{h}')
    m.optimize()
    return m
m = solve_problem()