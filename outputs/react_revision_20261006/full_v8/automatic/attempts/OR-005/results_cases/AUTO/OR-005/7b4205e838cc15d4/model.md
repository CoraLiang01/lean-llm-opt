Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups (demands), as defined by the current data.

Decision Variables:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from supplier $i$ to customer $j$.

Parameters:
- $c_{ij}$: unit transportation cost from supplier $i$ to customer $j$ (from file_2_view_0, columns demand1–demand8, rows supply1–supply8)
- $d_j$: demand of customer $j$ (from file_0_view_0, column demand)
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column supply_capacity)

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

Index sets:
- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

Data Mapping

- $I$ (suppliers): file_1_view_0, column "Supplier", all rows
- $J$ (customers): file_0_view_0, column "Customers", all rows
- $d_j$: file_0_view_0, column "demand", indexed by "Customers"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Supplier"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" (supplier, mapped as per row_id_mapping), columns demand1–demand8 (customer), all rows

Variable domain: $x_{ij} \geq 0$, continuous, for all $i \in I$, $j \in J$.