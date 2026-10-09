##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from distribution center (supplier) $i$ to customer group $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand of customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of distribution center $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** For each customer group $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:** For each distribution center $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:** For all $i \in I$, $j \in J$,
   $$
   x_{ij} \geq 0
   $$

##### Data Mapping

- $I$ (distribution centers/suppliers): supplier_id column in supply_capacity.csv and transportation_costs.csv
- $J$ (customer groups/demands): customer_id column in customer_demand.csv and demand columns in transportation_costs.csv
- $d_j$: demand column in customer_demand.csv, indexed by customer_id
- $s_i$: supply_capacity column in supply_capacity.csv, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_demand* columns in transportation_costs.csv, with row index supplier_id and column index demand* (see relationships in the Observation for exact mapping)

All indices, parameters, and coefficients are to be taken exactly as listed in the source files and relationships.