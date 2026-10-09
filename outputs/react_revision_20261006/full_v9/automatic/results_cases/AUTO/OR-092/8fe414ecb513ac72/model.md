## Mathematical Model

**Sets:**
- $S$: set of sources, indexed by $i$ (from expanded_sources.csv), $S = \{\text{S1}, \ldots, \text{S10}\}$
- $D$: set of destinations, indexed by $j$ (from expanded_destinations.csv), $D = \{\text{D1}, \ldots, \text{D20}\}$

**Parameters:**
- $c_{ij}$: unit transportation cost from source $i$ to destination $j$ (from expanded_cost_matrix.csv, table_id: file_0_view_0, columns: D1-D20, row: source_id)
- $a_i$: supply at source $i$ (from expanded_sources.csv, table_id: file_2_view_0, column: supply_units)
- $b_j$: demand at destination $j$ (from expanded_destinations.csv, table_id: file_1_view_0, column: demand_units)
- $Q$: truck capacity (units per truck), $Q = 10$

**Decision Variables:**
- $t_{ij} \in \mathbb{Z}_+ $: number of trucks dispatched from source $i$ to destination $j$ (integer, $\geq 0$)
- $x_{ij} \in [0, Q]$: cargo units shipped from $i$ to $j$ in trucks (continuous, $0 \leq x_{ij} \leq Q t_{ij}$)

**Objective:**
\[
\min \sum_{i \in S} \sum_{j \in D} c_{ij} x_{ij}
\]

**Constraints:**

1. **Supply constraints (do not exceed supply at each source):**
\[
\sum_{j \in D} x_{ij} \leq a_i \quad \forall i \in S
\]

2. **Demand constraints (meet demand at each destination):**
\[
\sum_{i \in S} x_{ij} = b_j \quad \forall j \in D
\]

3. **Truck loading constraints (cargo per route cannot exceed total truck capacity):**
\[
x_{ij} \leq Q \cdot t_{ij} \quad \forall i \in S,\, j \in D
\]

4. **Truck dispatch integrality:**
\[
t_{ij} \in \mathbb{Z}_+, \quad \forall i \in S,\, j \in D
\]

5. **Cargo nonnegativity and upper bound:**
\[
0 \leq x_{ij} \leq Q \cdot t_{ij} \quad \forall i \in S,\, j \in D
\]

---

**Data Mapping:**

- $S$ (sources): All source_id in expanded_sources.csv (file_2_view_0)
- $D$ (destinations): All destination_id in expanded_destinations.csv (file_1_view_0)
- $c_{ij}$: Entry in expanded_cost_matrix.csv (file_0_view_0), row source_id $i$, column $j$
- $a_i$: supply_units in expanded_sources.csv (file_2_view_0), row source_id $i$
- $b_j$: demand_units in expanded_destinations.csv (file_1_view_0), row destination_id $j$
- $Q$: 10 (truck capacity, from user description)
- $t_{ij}$: integer variable, number of trucks from $i$ to $j$
- $x_{ij}$: continuous variable, units shipped from $i$ to $j$ (partial truckloads allowed, but only if a truck is dispatched)

---

**Notes:**
- Each $t_{ij}$ is integer and represents the number of trucks dispatched on route $(i,j)$.
- Each $x_{ij}$ is continuous, $0 \leq x_{ij} \leq Q t_{ij}$, and represents the total units shipped from $i$ to $j$.
- All demand must be met exactly, and no source can ship more than its available supply.
- Costs are per unit shipped, not per truck.

---

**Summary:**  
This is a capacitated transportation problem with integer truck dispatches and partial truckloads allowed, minimizing total per-unit transportation cost, with all data and indices mapped directly to the provided CSV files.