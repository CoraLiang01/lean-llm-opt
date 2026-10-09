##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: set of suppliers, from file_1_view_0 column "Supplier"
- $J$: set of customers, from file_0_view_0 column "Customers"

##### Parameters

- $d_j$: demand of customer $j$, from file_0_view_0 column "demand"
- $s_i$: supply capacity of supplier $i$, from file_1_view_0 column "supply_capacity"
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from file_2_view_0, row "Unnamed: 0" (supplier), columns "demand1"..."demand8" (customers)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

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

##### Data Mapping

- $I$ (suppliers): all values in file_1_view_0 column "Supplier"
- $J$ (customers): all values in file_0_view_0 column "Customers"
- $d_j$: file_0_view_0, column "demand", indexed by "Customers"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Supplier"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" (supplier), columns "demand1"..."demand8" (customers), with supplier row IDs mapped as per row_id_mapping in the Observation

All indices, parameters, and coefficients are to be taken directly from the referenced columns and rows in the returned tables.