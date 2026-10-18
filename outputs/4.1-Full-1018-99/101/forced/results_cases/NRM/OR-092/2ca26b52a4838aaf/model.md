#### Abstract Model

**Index Sets:**
- $S$: set of source locations (from `expanded_sources.csv`, column `source_id`)
- $D$: set of demand locations (from `expanded_destinations.csv`, column `destination_id`)

**Parameters:**
- $s_i$: supply at source $i \in S$ (from `expanded_sources.csv`, column `supply_units`)
- $d_j$: demand at destination $j \in D$ (from `expanded_destinations.csv`, column `demand_units`)
- $c_{ij}$: unit transportation cost from source $i$ to destination $j$ (from `expanded_cost_matrix.csv`, column for $j$ in row $i$)
- $Q$: truck capacity (fixed at 10 units per truck)

**Decision Variables:**
- $x_{ij} \geq 0$: amount of cargo shipped from source $i$ to destination $j$ (continuous, non-negative)
- $t_{ij} \in \mathbb{Z}_+$: number of trucks dispatched from source $i$ to destination $j$ (integer, non-negative)

**Objective:**
\[
\min \sum_{i \in S} \sum_{j \in D} c_{ij} \cdot x_{ij}
\]

**Constraints:**

1. **Supply constraints (do not exceed supply at each source):**
   \[
   \sum_{j \in D} x_{ij} \leq s_i \quad \forall i \in S
   \]

2. **Demand constraints (meet demand at each destination):**
   \[
   \sum_{i \in S} x_{ij} = d_j \quad \forall j \in D
   \]

3. **Truck loading constraints (cargo per route cannot exceed total truck capacity):**
   \[
   x_{ij} \leq Q \cdot t_{ij} \quad \forall i \in S,\, j \in D
   \]

4. **Truck dispatch integrality:**
   \[
   t_{ij} \in \mathbb{Z}_+, \quad x_{ij} \geq 0 \quad \forall i \in S,\, j \in D
   \]

**Data Mapping:**

- $S$ and $s_i$ from `expanded_sources.csv` (`source_id`, `supply_units`)
- $D$ and $d_j$ from `expanded_destinations.csv` (`destination_id`, `demand_units`)
- $c_{ij}$ from `expanded_cost_matrix.csv` (`source_id`, columns `D1`–`D20`)
- $Q$ is a fixed constant (10 units/truck, from the user query)
- $x_{ij}$, $t_{ij}$ are decision variables as defined above

**Note:** All data is used as returned by the query, with no additional filtering or aggregation.