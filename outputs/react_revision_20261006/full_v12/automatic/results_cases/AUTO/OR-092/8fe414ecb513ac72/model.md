## Mathematical Model

**Sets:**
- $I$: set of sources, $I = \{\text{S1}, \ldots, \text{S10}\}$ (from expanded_sources.csv)
- $J$: set of destinations, $J = \{\text{D1}, \ldots, \text{D20}\}$ (from expanded_destinations.csv)

**Parameters:**
- $c_{ij}$: unit transportation cost from source $i$ to destination $j$ (from expanded_cost_matrix.csv, table_id: file_0_view_0, columns: source_id, D1–D20)
- $s_i$: supply at source $i$ (units) (from expanded_sources.csv, table_id: file_2_view_0, columns: source_id, supply_units)
- $d_j$: demand at destination $j$ (units) (from expanded_destinations.csv, table_id: file_1_view_0, columns: destination_id, demand_units)
- $Q$: truck capacity (units), $Q = 10$

**Decision Variables:**
- $t_{ij} \in \mathbb{Z}_+ $: number of trucks dispatched from source $i$ to destination $j$ (integer, $\geq 0$)
- $x_{ij} \in [0, Q]$: cargo (units) shipped from $i$ to $j$ in trucks, $x_{ij} = t_{ij} \cdot q_{ij}$, where $0 \leq q_{ij} \leq Q$ is the load per truck (partial loading allowed)

**Model:**

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} \, x_{ij}
$$

Subject to:

1. **Demand satisfaction at each destination:**
$$
\sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
$$

2. **Supply limit at each source:**
$$
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
$$

3. **Truck loading and integer trips:**
$$
0 \leq x_{ij} \leq Q \cdot t_{ij} \qquad \forall i \in I,\, j \in J
$$
$$
t_{ij} \in \mathbb{Z}_+ \qquad \forall i \in I,\, j \in J
$$

4. **Partial loading allowed:**
$$
0 \leq x_{ij} \leq Q \qquad \text{if } t_{ij} = 1
$$
$$
0 \leq x_{ij} \leq Q \cdot t_{ij} \qquad \forall t_{ij} \geq 0
$$

**Data Mapping:**
- $I$ from expanded_sources.csv, column source_id, table_id: file_2_view_0
- $J$ from expanded_destinations.csv, column destination_id, table_id: file_1_view_0
- $c_{ij}$ from expanded_cost_matrix.csv, table_id: file_0_view_0, columns source_id, D1–D20
- $s_i$ from expanded_sources.csv, column supply_units, table_id: file_2_view_0
- $d_j$ from expanded_destinations.csv, column demand_units, table_id: file_1_view_0

**Notes:**
- Each $t_{ij}$ is integer, $x_{ij}$ is continuous and $0 \leq x_{ij} \leq Q \cdot t_{ij}$.
- Partial loading: $x_{ij}$ can be any value in $[0, Q \cdot t_{ij}]$.
- All demands must be exactly met, all supplies not exceeded, and all truck trips are integer.

**Summary:**  
This is a capacitated transportation problem with integer truck trips, partial loading, fixed truck capacity, and per-unit route costs. All data and sets are mapped directly from the provided CSVs.