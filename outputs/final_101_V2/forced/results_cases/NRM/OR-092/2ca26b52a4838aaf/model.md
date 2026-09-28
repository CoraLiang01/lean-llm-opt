#### Index Sets

- $S$: set of source locations (from expanded_sources.csv, column source_id)
- $D$: set of demand locations (from expanded_destinations.csv, column destination_id)

#### Parameters

- $s_i$: supply at source $i \in S$ (from expanded_sources.csv, column supply_units)
- $d_j$: demand at destination $j \in D$ (from expanded_destinations.csv, column demand_units)
- $c_{ij}$: unit transportation cost from source $i$ to destination $j$ (from expanded_cost_matrix.csv, entry at row source_id $i$, column $j$)
- $Q$: truck capacity (fixed at 10 units per truck)

#### Decision Variables

- $x_{ij} \geq 0$: amount of cargo shipped from source $i$ to destination $j$
- $t_{ij} \in \mathbb{Z}_+$: number of trucks dispatched from source $i$ to destination $j$

#### Objective

$$
\min \sum_{i \in S} \sum_{j \in D} c_{ij} \cdot x_{ij}
$$

#### Constraints

1. **Supply constraints (at each source):**
   $$
   \sum_{j \in D} x_{ij} \leq s_i \quad \forall i \in S
   $$

2. **Demand constraints (at each destination):**
   $$
   \sum_{i \in S} x_{ij} = d_j \quad \forall j \in D
   $$

3. **Truck loading constraints (partial loading allowed, integer trucks):**
   $$
   x_{ij} \leq Q \cdot t_{ij} \quad \forall i \in S,\, j \in D
   $$

4. **Variable domains:**
   $$
   x_{ij} \geq 0 \quad \forall i \in S,\, j \in D
   $$
   $$
   t_{ij} \in \mathbb{Z}_+ \quad \forall i \in S,\, j \in D
   $$

---

#### Data Mapping

- **expanded_sources.csv**: source_id $\rightarrow$ $S$, supply_units $\rightarrow$ $s_i$
- **expanded_destinations.csv**: destination_id $\rightarrow$ $D$, demand_units $\rightarrow$ $d_j$
- **expanded_cost_matrix.csv**: source_id $\rightarrow$ $S$, columns $D1$-$D20$ $\rightarrow$ $D$, entries $\rightarrow$ $c_{ij}$

All sets, parameters, and relationships are defined symbolically and mapped to their exact table and column names as required.