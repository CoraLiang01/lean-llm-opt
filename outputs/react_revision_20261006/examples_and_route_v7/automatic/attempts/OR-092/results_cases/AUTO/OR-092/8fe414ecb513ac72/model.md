##### Decision Variables

Let $t_{ij} \in \mathbb{Z}_{\geq 0}$ be the number of trucks dispatched from source $i \in I$ to destination $j \in J$.

Let $x_{ij} \in [0,10]$ be the cargo (units) shipped from source $i$ to destination $j$ in trucks, with $x_{ij} \leq 10 t_{ij}$ and $x_{ij} \geq 0$.

##### Parameters

- $I = \{$S1, S2, ..., S10$\}$ (source_id from expanded_sources.csv)
- $J = \{$D1, D2, ..., D20$\}$ (destination_id from expanded_destinations.csv)
- $c_{ij}$: unit transportation cost from $i$ to $j$ (from expanded_cost_matrix.csv)
- $s_i$: supply at source $i$ (supply_units from expanded_sources.csv)
- $d_j$: demand at destination $j$ (demand_units from expanded_destinations.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each destination's demand must be met):
   $$
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   $$

2. **Supply limit** (each source cannot ship more than its supply):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$

3. **Truck loading and integer trips** (each route's cargo is at most 10 units per truck, and number of trucks is integer):
   $$
   0 \leq x_{ij} \leq 10 t_{ij} \quad \forall i \in I,\, j \in J
   $$
   $$
   t_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   $$

4. **Nonnegativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (sources): source_id from expanded_sources.csv and expanded_cost_matrix.csv rows
- $J$ (destinations): destination_id from expanded_destinations.csv and expanded_cost_matrix.csv columns
- $c_{ij}$: entry in expanded_cost_matrix.csv at row source_id $i$, column $j$
- $s_i$: supply_units in expanded_sources.csv for source_id $i$
- $d_j$: demand_units in expanded_destinations.csv for destination_id $j$