#### Mathematical Model

Let:
- $N$ = set of all nodes (sources, hubs, customers)
- $A$ = set of directed arcs $(i,j)$ with cost $c_{ij}$ (from arc_costs.csv)
- $S$ = set of source nodes = $\{\text{S1}, \text{S2}, \text{S3}\}$
- $H$ = set of hub nodes = $\{\text{H1}, \text{H2}\}$
- $C$ = set of customer nodes = $\{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

Decision variables:
- $f_{ij} \geq 0$: shipment flow on arc $(i,j) \in A$

Parameters:
- $s_i$: supply at source $i \in S$ (from node_supply_demand.csv)
- $d_j$: demand at customer $j \in C$ (from node_supply_demand.csv)
- $u_h$: throughput capacity at hub $h \in H$ (from hub_capacity.csv)
- $c_{ij}$: per-unit cost on arc $(i,j) \in A$ (from arc_costs.csv)

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
\sum_{i: (i,h)\in A} f_{ih} \leq u_h
$$

5. **Nonnegativity**:
$$
f_{ij} \geq 0 \quad \forall (i,j)\in A
$$

#### Data Mapping

- **Source supply and customer demand** (file_0_view_0, node_supply_demand.csv):
    - $s_{\text{S1}} = 120$, $s_{\text{S2}} = 100$, $s_{\text{S3}} = 90$
    - $d_{\text{C1}} = 70$, $d_{\text{C2}} = 80$, $d_{\text{C3}} = 60$, $d_{\text{C4}} = 90$
- **Hub throughput capacity** (file_1_view_0, hub_capacity.csv):
    - $u_{\text{H1}} = 170$, $u_{\text{H2}} = 160$
- **Arc costs** (file_2_view_0, arc_costs.csv):
    - $A = \{(\text{S1},\text{H1}), (\text{S1},\text{H2}), (\text{S2},\text{H1}), (\text{S2},\text{H2}), (\text{S3},\text{H1}), (\text{S3},\text{H2}), (\text{H1},\text{C1}), (\text{H1},\text{C2}), (\text{H1},\text{C3}), (\text{H1},\text{C4}), (\text{H2},\text{C1}), (\text{H2},\text{C2}), (\text{H2},\text{C3}), (\text{H2},\text{C4})\}$
    - $c_{ij}$ as given in arc_costs.csv for each arc $(i,j)$

All indices, parameters, and constraints are mapped directly from the retrieved data.