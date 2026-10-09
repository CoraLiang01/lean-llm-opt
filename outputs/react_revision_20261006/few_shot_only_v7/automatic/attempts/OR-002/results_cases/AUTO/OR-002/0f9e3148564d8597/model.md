##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from store $i \in I$ to customer group $j \in J$ (continuous).

##### Sets

$I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$ (stores)  
$J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$ (customer groups)

##### Parameters

- $d_j$: demand of customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of store $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from store $i$ to customer group $j$ (from transportation_costs.csv)

##### Objective Function

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

- $d_j$:  
  From "customer_demand.csv", column "customer" gives $j$, column "demand" gives $d_j$.

- $s_i$:  
  From "supply_capacity.csv", column "Unnamed: 0" gives $i$, column "supply_capacity" gives $s_i$.

- $c_{ij}$:  
  From "transportation_costs.csv", row "Unnamed: 0" gives $i$, columns "C1"–"C12" give $c_{ij}$ for each $j$.

All identifiers and coefficients are to be used exactly as in the source data.