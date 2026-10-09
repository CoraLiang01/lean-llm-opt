#### Mathematical Model

Let:
- $N$ = set of all nodes (sources, hubs, customers)
- $S$ = set of source nodes = $\{\text{S1}, \text{S2}, \text{S3}\}$
- $H$ = set of hub nodes = $\{\text{H1}, \text{H2}\}$
- $C$ = set of customer nodes = $\{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$
- $A$ = set of directed arcs $(i,j)$ with cost $c_{ij}$ as listed in arc_costs.csv
- $f_{ij} \geq 0$ = shipment flow on arc $(i,j) \in A$

Parameters (from data):
- Source supply: $s_i$ for $i \in S$ (from node_supply_demand.csv, NodeType = SourceSupply)
- Customer demand: $d_j$ for $j \in C$ (from node_supply_demand.csv, NodeType = CustomerDemand)
- Hub throughput capacity: $u_h$ for $h \in H$ (from hub_capacity.csv)
- Arc cost: $c_{ij}$ for $(i,j) \in A$ (from arc_costs.csv)

Objective:
$$
\min \sum_{(i,j)\in A} c_{ij} f_{ij}
$$

Subject to:

1. Source supply upper bounds:
$$
\sum_{j: (i,j)\in A} f_{ij} \leq s_i \quad \forall i \in S
$$

2. Customer demand constraints:
$$
\sum_{i: (i,j)\in A} f_{ij} \geq d_j \quad \forall j \in C
$$

3. Flow balance at each hub:
$$
\sum_{i: (i,h)\in A} f_{ih} = \sum_{j: (h,j)\in A} f_{hj} \quad \forall h \in H
$$

4. Hub throughput capacity constraints:
$$
\sum_{i: (i,h)\in A} f_{ih} \leq u_h \quad \forall h \in H
$$

5. Nonnegativity:
$$
f_{ij} \geq 0 \quad \forall (i,j) \in A
$$

#### Data Mapping

- Source nodes $S$: S1, S2, S3 (file_0_view_0, NodeType = SourceSupply)
    - $s_{\text{S1}} = 120$
    - $s_{\text{S2}} = 100$
    - $s_{\text{S3}} = 90$
- Customer nodes $C$: C1, C2, C3, C4 (file_0_view_0, NodeType = CustomerDemand)
    - $d_{\text{C1}} = 70$
    - $d_{\text{C2}} = 80$
    - $d_{\text{C3}} = 60$
    - $d_{\text{C4}} = 90$
- Hub nodes $H$: H1, H2 (file_1_view_0, Hub)
    - $u_{\text{H1}} = 170$
    - $u_{\text{H2}} = 160$
- Arcs $A$ and costs $c_{ij}$: all (From, To, Cost) in file_2_view_0 (arc_costs.csv)
    - Each $(i,j)$ with $c_{ij}$ as listed in the table

- Variables: $f_{ij}$ for each arc $(i,j)$ in $A$ (file_2_view_0, From, To)

All sets, parameters, and constraints are defined directly from the retrieved CSV data, preserving all identifiers and coefficients.