##### Mathematical Model

Let:
- $N$ = set of all nodes (sources, hubs, customers)
- $S$ = set of source nodes (from node_supply_demand.csv where NodeType = SourceSupply)
- $H$ = set of hub nodes (from hub_capacity.csv)
- $C$ = set of customer nodes (from node_supply_demand.csv where NodeType = CustomerDemand)
- $A$ = set of directed arcs $(i,j)$ with cost $c_{ij}$ (from arc_costs.csv)
- $f_{ij} \geq 0$ = shipment flow on arc $(i,j) \in A$

Parameters:
- $s_i$ = supply at source $i \in S$ (from node_supply_demand.csv)
- $d_j$ = demand at customer $j \in C$ (from node_supply_demand.csv)
- $u_h$ = throughput capacity at hub $h \in H$ (from hub_capacity.csv)
- $c_{ij}$ = per-unit cost on arc $(i,j) \in A$ (from arc_costs.csv)

Objective:
$$
\min \sum_{(i,j)\in A} c_{ij} f_{ij}
$$

Subject to:

1. **Source supply upper bounds** (for all $i \in S$):
$$
\sum_{j: (i,j)\in A} f_{ij} \leq s_i
$$

2. **Customer demand constraints** (for all $j \in C$):
$$
\sum_{i: (i,j)\in A} f_{ij} \geq d_j
$$

3. **Flow balance at each hub** (for all $h \in H$):
$$
\sum_{i: (i,h)\in A} f_{ih} = \sum_{j: (h,j)\in A} f_{hj}
$$

4. **Hub throughput capacity** (for all $h \in H$):
$$
\sum_{i: (i,h)\in A} f_{ih} + \sum_{j: (h,j)\in A} f_{hj} \leq u_h
$$

5. **Nonnegativity**:
$$
f_{ij} \geq 0 \quad \forall (i,j)\in A
$$

##### Data Mapping

- **Source nodes $S$ and their supplies $s_i$**: file_0_view_0, column NodeType = "SourceSupply", Node, Amount
- **Customer nodes $C$ and their demands $d_j$**: file_0_view_0, column NodeType = "CustomerDemand", Node, Amount
- **Hub nodes $H$ and their capacities $u_h$**: file_1_view_0, columns Hub, ThroughputCapacity
- **Arcs $A$ and costs $c_{ij}$**: file_2_view_0, columns From, To, Cost
- **Variables $f_{ij}$**: defined for all $(i,j)$ in $A$ (all rows of file_2_view_0)

All index sets and parameters are defined exactly as listed in the current CSV files, preserving all node and arc identifiers.