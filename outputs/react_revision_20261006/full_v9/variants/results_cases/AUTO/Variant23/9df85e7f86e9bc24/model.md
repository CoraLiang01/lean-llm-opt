## Mathematical Model

**Sets:**
- $I$: set of depots (from depot_capacity.csv), $I = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}\}$
- $J$: set of markets (from market_demand.csv), $J = \{\text{M1}, \text{M2}, \text{M3}, \text{M4}, \text{M5}\}$

**Parameters:**
- $S_i$: supply capacity of depot $i \in I$ (from depot_capacity.csv, column SupplyCapacity, table_id: file_0_view_0)
- $D_j$: demand of market $j \in J$ (from market_demand.csv, column Demand, table_id: file_1_view_0)
- $c_{ij}$: variable shipping cost per unit from depot $i$ to market $j$ (from route_variable_costs.csv, table_id: file_2_view_0, columns Depot, $J$)
- $f_{ij}$: fixed activation cost for route $i$-$j$ (from route_fixed_costs.csv, table_id: file_3_view_0, columns Depot, $J$)
- $M_{ij} = \min(S_i, D_j)$: maximum possible shipment on route $i$-$j$ (computed from $S_i$ and $D_j$ above)

**Decision Variables:**
- $x_{ij} \geq 0$: quantity shipped from depot $i$ to market $j$
- $y_{ij} \in \{0,1\}$: 1 if route $i$-$j$ is activated, 0 otherwise

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} \left( c_{ij} x_{ij} + f_{ij} y_{ij} \right)
\]

**Subject to:**

1. **Market demand constraints:**
   \[
   \sum_{i \in I} x_{ij} \geq D_j \qquad \forall j \in J
   \]

2. **Depot supply constraints:**
   \[
   \sum_{j \in J} x_{ij} \leq S_i \qquad \forall i \in I
   \]

3. **Route activation linking constraints:**
   \[
   x_{ij} \leq M_{ij} \, y_{ij} \qquad \forall i \in I,\, j \in J
   \]
   where $M_{ij} = \min(S_i, D_j)$

4. **Nonnegativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

5. **Binary route activation:**
   \[
   y_{ij} \in \{0,1\} \qquad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- **Depots $I$**: file_0_view_0, column Depot
- **Markets $J$**: file_1_view_0, column Market
- **$S_i$**: file_0_view_0, column SupplyCapacity
- **$D_j$**: file_1_view_0, column Demand
- **$c_{ij}$**: file_2_view_0, row Depot $i$, column $j$
- **$f_{ij}$**: file_3_view_0, row Depot $i$, column $j$
- **$M_{ij}$**: $\min(S_i, D_j)$, computed from above

All indices, parameters, and constraints are defined exactly as required by the current data and user query.