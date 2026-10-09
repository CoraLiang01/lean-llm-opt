##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

##### Parameters

- $d_j$: demand at outlet $j$ (from customer_demand.csv)
- $s_i$: supply capacity at plant $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each outlet's demand must be met:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:** Each plant's shipments cannot exceed its capacity:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $d_j$ from table_id=file_0_view_0, column=customer (ID), column=demand (value)
- $s_i$ from table_id=file_1_view_0, column=Unnamed: 0 (ID), column=supply_capacity (value)
- $c_{ij}$ from table_id=file_2_view_0, row=Unnamed: 0 (plant ID), columns=C1, C2, C3, C4 (outlet IDs)