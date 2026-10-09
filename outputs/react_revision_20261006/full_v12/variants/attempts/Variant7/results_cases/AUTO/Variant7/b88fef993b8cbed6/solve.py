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
             'filters': {'conditions': [{'column': 'NodeType',
                                         'dtype': 'string',
                                         'evidence': 'source supplies and customer demands are listed in '
                                                     'node_supply_demand.csv',
                                         'operator': 'in',
                                         'value': ['SourceSupply', 'CustomerDemand']}],
                         'logic': 'or'},
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
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Hub': 'H1', 'ThroughputCapacity': '170'}},
                         {'source_row': 1, 'values': {'Hub': 'H2', 'ThroughputCapacity': '160'}}],
             'returned_rows': 2,
             'role': 'hub throughput capacity',
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
             'role': 'arc transportation costs',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    node_df = CSVQA_FRAMES['file_0_view_0']
    hub_df = CSVQA_FRAMES['file_1_view_0']
    arc_df = CSVQA_FRAMES['file_2_view_0']
    sources = []
    source_supply = {}
    customers = []
    customer_demand = {}
    for (_, row) in node_df.iterrows():
        node = row['Node']
        ntype = row['NodeType']
        amount = float(row['Amount'])
        if str(ntype).casefold() == 'sourcesupply':
            sources.append(node)
            source_supply[node] = amount
        elif str(ntype).casefold() == 'customerdemand':
            customers.append(node)
            customer_demand[node] = amount
    hubs = []
    hub_capacity = {}
    for (_, row) in hub_df.iterrows():
        hub = row['Hub']
        cap = float(row['ThroughputCapacity'])
        hubs.append(hub)
        hub_capacity[hub] = cap
    arcs = []
    arc_cost = {}
    for (_, row) in arc_df.iterrows():
        i = row['From']
        j = row['To']
        c = float(row['Cost'])
        arcs.append((i, j))
        arc_cost[i, j] = c
    outgoing_arcs = {}
    incoming_arcs = {}
    for (i, j) in arcs:
        outgoing_arcs.setdefault(i, []).append((i, j))
        incoming_arcs.setdefault(j, []).append((i, j))
    m = gp.Model('min_cost_transshipment')
    f_vars = m.addVars(arcs, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((arc_cost[i, j] * f_vars[i, j] for (i, j) in arcs)), GRB.MINIMIZE)
    for s in sources:
        if s in outgoing_arcs:
            m.addConstr(gp.quicksum((f_vars[i, j] for (i, j) in outgoing_arcs[s])) <= source_supply[s], name='supply_' + s)
        else:
            m.addConstr(0 <= source_supply[s], name='supply_' + s)
    for c in customers:
        if c in incoming_arcs:
            m.addConstr(gp.quicksum((f_vars[i, j] for (i, j) in incoming_arcs[c])) >= customer_demand[c], name='demand_' + c)
        else:
            m.addConstr(0 >= customer_demand[c], name='demand_' + c)
    for h in hubs:
        in_expr = gp.quicksum((f_vars[i, j] for (i, j) in incoming_arcs.get(h, [])))
        out_expr = gp.quicksum((f_vars[i, j] for (i, j) in outgoing_arcs.get(h, [])))
        m.addConstr(in_expr == out_expr, name='balance_' + h)
    for h in hubs:
        in_expr = gp.quicksum((f_vars[i, j] for (i, j) in incoming_arcs.get(h, [])))
        m.addConstr(in_expr <= hub_capacity[h], name='hubcap_' + h)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)