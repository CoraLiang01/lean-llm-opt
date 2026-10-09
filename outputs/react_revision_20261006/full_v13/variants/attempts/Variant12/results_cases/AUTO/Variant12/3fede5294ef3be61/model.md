## Symbolic Mathematical Model

**Sets:**
- $I$: set of plants (from plant_capacity.csv), $I = \{\text{P1}, \text{P2}, \text{P3}\}$
- $J$: set of retailers (from retailer_demand.csv), $J = \{\text{R1}, \text{R2}, \text{R3}, \text{R4}, \text{R5}, \text{R6}\}$

**Parameters:**
- $S_i$: supply capacity at plant $i \in I$ (from plant_capacity.csv, column SupplyCapacity, table_id file_0_view_0)
- $D_j$: demand at retailer $j \in J$ (from retailer_demand.csv, column Demand, table_id file_1_view_0)
- $c_{ij}$: variable cost per carton from plant $i$ to retailer $j$ (from route_variable_costs.csv, table_id file_2_view_0, row Plant $i$, column $j$)
- $f_{ij}$: fixed cost to open route $i$-$j$ (from route_fixed_costs.csv, table_id file_3_view_0, row Plant $i$, column $j$)
- $M_{ij} = \min(S_i, D_j)$: maximum possible shipment on route $i$-$j$ (computed from S and D above)

**Decision Variables:**
- $x_{ij} \geq 0$: cartons shipped from plant $i$ to retailer $j$ (continuous, nonnegative)
- $y_{ij} \in \{0,1\}$: 1 if route $i$-$j$ is opened, 0 otherwise (binary)

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} \left( c_{ij} x_{ij} + f_{ij} y_{ij} \right)
\]

**Subject to:**

1. **Retailer demand constraints:**
   \[
   \sum_{i \in I} x_{ij} = D_j \qquad \forall j \in J
   \]

2. **Plant supply upper-bound constraints:**
   \[
   \sum_{j \in J} x_{ij} \leq S_i \qquad \forall i \in I
   \]

3. **Shipment-to-route activation linking constraints:**
   \[
   x_{ij} \leq M_{ij} \, y_{ij} \qquad \forall i \in I,\, j \in J
   \]
   where $M_{ij} = \min(S_i, D_j)$

4. **Nonnegativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

5. **Binary restrictions:**
   \[
   y_{ij} \in \{0,1\} \qquad \forall i \in I,\, j \in J
   \]

---

## Data Mapping

- **Plants ($I$):** All rows in plant_capacity.csv, column Plant, table_id file_0_view_0
- **Retailers ($J$):** All rows in retailer_demand.csv, column Retailer, table_id file_1_view_0
- **$S_i$:** plant_capacity.csv, column SupplyCapacity, table_id file_0_view_0
- **$D_j$:** retailer_demand.csv, column Demand, table_id file_1_view_0
- **$c_{ij}$:** route_variable_costs.csv, table_id file_2_view_0, row Plant $i$, column $j$
- **$f_{ij}$:** route_fixed_costs.csv, table_id file_3_view_0, row Plant $i$, column $j$
- **$M_{ij}$:** computed as $\min(S_i, D_j)$ for each $(i,j)$ pair

- **$x_{ij}$:** shipment quantity from plant $i$ to retailer $j$ (decision variable)
- **$y_{ij}$:** binary route activation variable for plant $i$ to retailer $j$ (decision variable)