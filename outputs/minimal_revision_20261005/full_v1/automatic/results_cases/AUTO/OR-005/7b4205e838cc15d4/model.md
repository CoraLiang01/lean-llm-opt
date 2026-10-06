##### Symbolic Model

Let $I$ be the set of suppliers (distribution centers) and $J$ the set of customer groups:
- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer group $j \in J$.

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
- Demand satisfaction: $\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J$
- Supply capacity:   $\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I$
- Non-negativity:    $x_{ij} \geq 0$ for all $i \in I$, $j \in J$

##### Data Mapping

- $d_j$: demand for customer group $j$ from file_0_view_0, column "demand", row with "Customers" = $j$
- $s_i$: supply capacity for supplier $i$ from file_1_view_0, column "supply_capacity", row with "Supplier" = $i$
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ from file_2_view_0, row with "Unnamed: 0" = $i$, column $j$

###### Table Bindings

- Customer demands: file_0_view_0
    - Columns: "Customers", "demand"
    - Rows: All (source_row 0–7)
- Supply capacities: file_1_view_0
    - Columns: "Supplier", "supply_capacity"
    - Rows: All (source_row 0–7)
- Transportation costs: file_2_view_0
    - Rows: All (source_row 0–7, "Unnamed: 0" = supplier)
    - Columns: "demand1", ..., "demand8" (customer group)

All identifiers and coefficients are to be taken directly from the corresponding table and column as described above. No data is omitted or aggregated.