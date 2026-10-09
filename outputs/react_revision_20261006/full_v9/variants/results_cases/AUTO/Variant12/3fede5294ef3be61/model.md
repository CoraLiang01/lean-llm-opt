## Symbolic Mathematical Model

**Sets:**
- $I$: set of plants (from plant_capacity.csv), $I = \{\text{P1}, \text{P2}, \text{P3}\}$
- $J$: set of retailers (from retailer_demand.csv), $J = \{\text{R1}, \text{R2}, \text{R3}, \text{R4}, \text{R5}, \text{R6}\}$

**Parameters:**
- $S_i$: supply capacity of plant $i \in I$ (from plant_capacity.csv, column SupplyCapacity, table_id file_0_view_0)
- $D_j$: demand of retailer $j \in J$ (from retailer_demand.csv, column Demand, table_id file_1_view_0)
- $c_{ij}$: variable cost per carton from plant $i$ to retailer $j$ (from route_variable_costs.csv, table_id file_2_view_0, columns Plant, $J$)
- $f_{ij}$: fixed cost to open route $i$-$j$ (from route_fixed_costs.csv, table_id file_3_view_0, columns Plant, $J$)
- $M_{ij} = \min(S_i, D_j)$: maximum possible shipment on route $i$-$j$ (computed from $S_i$ and $D_j$)

**Decision Variables:**
- $x_{ij} \geq 0$: cartons shipped from plant $i$ to retailer $j$ (continuous, nonnegative)
- $y_{ij} \in \{0,1\}$: 1 if route $i$-$j$ is opened, 0 otherwise (binary)

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} \left( c_{ij} x_{ij} + f_{ij} y_{ij} \right)
\]

**Constraints:**

1. **Retailer demand satisfaction:**
   \[
   \sum_{i \in I} x_{ij} = D_j \qquad \forall j \in J
   \]

2. **Plant supply capacity:**
   \[
   \sum_{j \in J} x_{ij} \leq S_i \qquad \forall i \in I
   \]

3. **Route activation linking:**
   \[
   x_{ij} \leq M_{ij} y_{ij} \qquad \forall i \in I,\, j \in J
   \]
   where $M_{ij} = \min(S_i, D_j)$

4. **Nonnegativity and integrality:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]
   \[
   y_{ij} \in \{0,1\} \qquad \forall i \in I,\, j \in J
   \]

---

## Data Mapping

- **Plants ($I$):** All Plant values from plant_capacity.csv (table_id file_0_view_0, column Plant)
- **Retailers ($J$):** All Retailer values from retailer_demand.csv (table_id file_1_view_0, column Retailer)
- **$S_i$:** SupplyCapacity from plant_capacity.csv (file_0_view_0, column SupplyCapacity) for each $i$
- **$D_j$:** Demand from retailer_demand.csv (file_1_view_0, column Demand) for each $j$
- **$c_{ij}$:** route_variable_costs.csv (file_2_view_0), entry at row Plant $i$, column $j$
- **$f_{ij}$:** route_fixed_costs.csv (file_3_view_0), entry at row Plant $i$, column $j$
- **$M_{ij}$:** $\min(S_i, D_j)$ for each $(i,j)$ pair

- **$x_{ij}$:** Nonnegative continuous variable for each $(i,j)$
- **$y_{ij}$:** Binary variable for each $(i,j)$

All sets, parameters, and variables are defined over the full set of entities present in the current CSV files.