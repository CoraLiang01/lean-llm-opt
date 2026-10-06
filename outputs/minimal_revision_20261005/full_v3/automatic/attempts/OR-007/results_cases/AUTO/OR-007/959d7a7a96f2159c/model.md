##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets and Indices

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses, from `region` in `supply_capacity.csv` and row_id_mapping in `transportation_costs.csv`)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores, from `customer` in `customer_demand.csv` and column_id_mapping in `transportation_costs.csv`)

##### Parameters

- $d_j$: demand of store $j$ (from `demand` in `customer_demand.csv`, table_id: file_0_view_0)
- $s_i$: supply capacity of warehouse $i$ (from `supply_capacity` in `supply_capacity.csv`, table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from `transportation_costs.csv`, table_id: file_2_view_0, row_id_mapping and column_id_mapping as above)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$
2. **Supply capacity:**  
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $d_j$ is from `customer_demand.csv` (table_id: file_0_view_0), column `demand`, indexed by `customer`.
- $s_i$ is from `supply_capacity.csv` (table_id: file_1_view_0), column `supply_capacity`, indexed by `region`.
- $c_{ij}$ is from `transportation_costs.csv` (table_id: file_2_view_0), with row_id_mapping:  
  S1, S2, S3, S4, S5 (from `Unnamed: 0`)  
  and column_id_mapping:  
  D1, D2, D3, D4, D5.

All indices and coefficients are bound to the supplied data and identifiers.