## Mathematical Model

**Sets:**
- $S$: set of sources, indexed by $i$ (from expanded_sources.csv), $S = \{\text{S1}, \ldots, \text{S10}\}$
- $D$: set of destinations, indexed by $j$ (from expanded_destinations.csv), $D = \{\text{D1}, \ldots, \text{D20}\}$

**Parameters:**
- $c_{ij}$: unit transportation cost from source $i$ to destination $j$ (from expanded_cost_matrix.csv, per unit cargo)
- $a_i$: supply at source $i$ (from expanded_sources.csv, in units)
- $b_j$: demand at destination $j$ (from expanded_destinations.csv, in units)
- $Q$: truck capacity (units per truck), $Q = 10$

**Decision Variables:**
- $t_{ij} \in \mathbb{Z}_+ $: number of trucks dispatched from source $i$ to destination $j$ (integer, $\geq 0$)
- $x_{ij} \in [0, Q]$: amount of cargo shipped from $i$ to $j$ (continuous, $0 \leq x_{ij} \leq Q t_{ij}$)

**Objective:**
\[
\min \sum_{i \in S} \sum_{j \in D} c_{ij} x_{ij}
\]

**Constraints:**

1. **Truck loading and integer trips:**
   \[
   0 \leq x_{ij} \leq Q t_{ij} \qquad \forall i \in S,\, j \in D
   \]
   \[
   t_{ij} \in \mathbb{Z}_+ \qquad \forall i \in S,\, j \in D
   \]

2. **Supply constraints:**
   \[
   \sum_{j \in D} x_{ij} \leq a_i \qquad \forall i \in S
   \]

3. **Demand constraints:**
   \[
   \sum_{i \in S} x_{ij} = b_j \qquad \forall j \in D
   \]

4. **Nonnegativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in S,\, j \in D
   \]

---

## Data Mapping

- $S$ (sources): All source_id in expanded_sources.csv and expanded_cost_matrix.csv rows.
- $D$ (destinations): All destination_id in expanded_destinations.csv and expanded_cost_matrix.csv columns.
- $c_{ij}$: Value in expanded_cost_matrix.csv at row source_id $i$, column $j$.
- $a_i$: supply_units in expanded_sources.csv for source_id $i$.
- $b_j$: demand_units in expanded_destinations.csv for destination_id $j$.
- $Q$: 10 (truck capacity, from problem statement).

**Variables:**
- $t_{ij}$: integer number of trucks from $i$ to $j$.
- $x_{ij}$: continuous cargo shipped from $i$ to $j$, $0 \leq x_{ij} \leq 10 t_{ij}$.

**All indices, parameters, and variables are defined over the full set of sources and destinations as listed in the current CSV files.**