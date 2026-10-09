##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$ (continuous).

##### Sets

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand of customer group $j \in J$ (from customer_demand.csv)
- $s_i$: supply capacity of distribution center $i \in I$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:**  
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (distribution centers/suppliers): all "Supplier" values in supply_capacity.csv and "Unnamed: 0" in transportation_costs.csv, matched by row.
- $J$ (customer groups): all "Customers" in customer_demand.csv and column headers in transportation_costs.csv, matched by column.
- $d_j$: "demand" column in customer_demand.csv, indexed by "Customers".
- $s_i$: "supply_capacity" column in supply_capacity.csv, indexed by "Supplier".
- $c_{ij}$: value in transportation_costs.csv at row with "Unnamed: 0" = $i$ and column $j$.

No data is omitted or aggregated; all identifiers and coefficients are preserved as in the source files.