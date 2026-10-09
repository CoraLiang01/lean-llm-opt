##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

##### Sets

$I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$

$J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Parameters

- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)

##### Objective

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. **Demand satisfaction:**  
   $\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J$

2. **Supply capacity:**  
   $\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I$

3. **Non-negativity:**  
   $x_{ij} \geq 0 \quad \forall i \in I,\, j \in J$

---

#### Data Mapping

- $i$ (supplier): "Unnamed: 0" column in supply_capacity.csv and transportation_costs.csv
- $j$ (customer): "customer" column in customer_demand.csv and column headers in transportation_costs.csv
- $d_j$: "demand" column in customer_demand.csv
- $s_i$: "supply_capacity" column in supply_capacity.csv
- $c_{ij}$: value in transportation_costs.csv at row $i$, column $j$