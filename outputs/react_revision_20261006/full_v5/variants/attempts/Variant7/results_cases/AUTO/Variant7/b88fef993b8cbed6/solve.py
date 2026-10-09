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
             'filters': {},
             'original_rows': 7,
             'records': [{'source_row': 0, 'values': {'Amount': '120', 'Node': 'S1', 'NodeType': 'SourceSupply'}},
                         {'source_row': 1, 'values': {'Amount': '100', 'Node': 'S2', 'NodeType': 'SourceSupply'}},
                         {'source_row': 2, 'values': {'Amount': '90', 'Node': 'S3', 'NodeType': 'SourceSupply'}},
                         {'source_row': 3, 'values': {'Amount': '70', 'Node': 'C1', 'NodeType': 'CustomerDemand'}},
                         {'source_row': 4, 'values': {'Amount': '80', 'Node': 'C2', 'NodeType': 'CustomerDemand'}},
                         {'source_row': 5, 'values': {'Amount': '60', 'Node': 'C3', 'NodeType': 'CustomerDemand'}},
                         {'source_row': 6, 'values': {'Amount': '90', 'Node': 'C4', 'NodeType': 'CustomerDemand'}}],
             'returned_rows': 7,
             'role': 'node supply and demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Hub', 'ThroughputCapacity'],
             'file_index': 1,
             'file_name': 'hub_capacity.csv',
             'filters': {},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Hub': 'H1', 'ThroughputCapacity': '170'}},
                         {'source_row': 1, 'values': {'Hub': 'H2', 'ThroughputCapacity': '160'}}],
             'returned_rows': 2,
             'role': 'hub throughput capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['From', 'To', 'Cost'],
             'file_index': 2,
             'file_name': 'arc_costs.csv',
             'filters': {},
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
             'role': 'arc transportation costs',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    node_supply_demand = [rec['values'] for rec in CSVQA_DATA['tables'][0]['records']]
    hub_capacity = [rec['values'] for rec in CSVQA_DATA['tables'][1]['records']]
    arc_costs = [rec['values'] for rec in CSVQA_DATA['tables'][2]['records']]
    sources = [rec['Node'] for rec in node_supply_demand if rec['NodeType'].casefold() == 'sourcesupply']
    customers = [rec['Node'] for rec in node_supply_demand if rec['NodeType'].casefold() == 'customerdemand']
    hubs = [rec['Hub'] for rec in hub_capacity]
    nodes = sources + hubs + customers
    supply = {rec['Node']: float(rec['Amount']) for rec in node_supply_demand if rec['NodeType'].casefold() == 'sourcesupply'}
    demand = {rec['Node']: float(rec['Amount']) for rec in node_supply_demand if rec['NodeType'].casefold() == 'customerdemand'}
    hub_cap = {rec['Hub']: float(rec['ThroughputCapacity']) for rec in hub_capacity}
    arcs = []
    cost = {}
    for rec in arc_costs:
        i = rec['From']
        j = rec['To']
        arcs.append((i, j))
        cost[i, j] = float(rec['Cost'])
    if len(cost) != len(arcs):
        raise ValueError('Mismatch in arc cost data.')
    m = gp.Model('DistributionNetworkTransshipment')
    f = m.addVars(arcs, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i, j] * f[i, j] for (i, j) in arcs)), GRB.MINIMIZE)
    for i in sources:
        out_arcs = [(i, j) for (ii, j) in arcs if ii == i]
        m.addConstr(gp.quicksum((f[i, j] for (i, j) in out_arcs)) <= supply[i], name=f'supply_{i}')
    for j in customers:
        in_arcs = [(i, j) for (i, jj) in arcs if jj == j]
        m.addConstr(gp.quicksum((f[i, j] for (i, j) in in_arcs)) >= demand[j], name=f'demand_{j}')
    for h in hubs:
        in_arcs = [(i, h) for (i, hh) in arcs if hh == h]
        out_arcs = [(h, j) for (hh, j) in arcs if hh == h]
        m.addConstr(gp.quicksum((f[i, h] for (i, h) in in_arcs)) == gp.quicksum((f[h, j] for (h, j) in out_arcs)), name=f'flowbal_{h}')
    for h in hubs:
        in_arcs = [(i, h) for (i, hh) in arcs if hh == h]
        m.addConstr(gp.quicksum((f[i, h] for (i, h) in in_arcs)) <= hub_cap[h], name=f'hubcap_{h}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')