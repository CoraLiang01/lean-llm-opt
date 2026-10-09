Mathematical Model (Transportation Problem for FreshMart):

Sets:
- Let S = {Supplier1, Supplier2, Supplier3, Supplier4, Supplier5} be the set of warehouses (indexed by s).
- Let C = {Customer1, Customer2, Customer3, Customer4, Customer5, Customer6} be the set of retail stores (indexed by c).

Parameters (from source data):
- demand_c: Demand at customer c. (file_0_view_0, column "demand", indexed by "Customers")
- supply_capacity_s: Supply capacity at supplier s. (file_1_view_0, column "supply_capacity", indexed by "Suppliers")
- cost_sc: Transportation cost per unit from supplier s to customer c. (file_2_view_0, columns "Customer1"..."Customer6", rows "Unnamed: 0" = supplier IDs)

Decision Variables:
- x_{s,c} ≥ 0: Amount of fresh produce shipped from supplier s to customer c (continuous, non-negative).

Objective:
Minimize total transportation cost:
\[
\min \sum_{s \in S} \sum_{c \in C} \text{cost}_{s,c} \cdot x_{s,c}
\]

Subject to:

1. Demand satisfaction at each customer:
\[
\sum_{s \in S} x_{s,c} = \text{demand}_c \quad \forall c \in C
\]

2. Supply capacity at each supplier:
\[
\sum_{c \in C} x_{s,c} \leq \text{supply\_capacity}_s \quad \forall s \in S
\]

3. Non-negativity:
\[
x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
\]

Data Mapping:
- S = all "Suppliers" in file_1_view_0 and "Unnamed: 0" in file_2_view_0.
- C = all "Customers" in file_0_view_0 and columns "Customer1"..."Customer6" in file_2_view_0.
- demand_c: file_0_view_0, column "demand", indexed by "Customers".
- supply_capacity_s: file_1_view_0, column "supply_capacity", indexed by "Suppliers".
- cost_sc: file_2_view_0, value at row "Unnamed: 0" = s, column = c.

Variable domain:
- x_{s,c} ∈ [0, ∞), ∀ s ∈ S, c ∈ C.

All indices, parameters, and coefficients are bound exactly to the current source data as described above.