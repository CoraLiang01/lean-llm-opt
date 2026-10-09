##### Mathematical Model

Let $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$ (warehouses)  
Let $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$ (stores)

Let $x_{ij} \geq 0$ be the amount shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Parameters:
- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each store:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity for each warehouse:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (warehouses): all "Suppliers" in supply_capacity.csv and transportation_costs.csv row labels
- $J$ (stores): all "Customers" in customer_demand.csv and transportation_costs.csv column labels
- $d_j$: "demand" column in customer_demand.csv, indexed by "Customers"
- $s_i$: "supply_capacity" column in supply_capacity.csv, indexed by "Suppliers"
- $c_{ij}$: entry in transportation_costs.csv at row "Unnamed: 0" = $i$, column = $j$