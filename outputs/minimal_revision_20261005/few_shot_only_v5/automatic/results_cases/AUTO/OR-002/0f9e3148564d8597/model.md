##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from store $i$ to customer group $j$.

- $i \in I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$
- $j \in J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Parameters

- $d_j$: demand of customer group $j$
- $s_i$: supply capacity of store $i$
- $c_{ij}$: transportation cost per unit from store $i$ to customer group $j$

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer group $j$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each store $i$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **Customer Demands** (from `customer_demand.csv`; column: `customer`, `demand`):  
  $d_{\text{C1}} = 11$  
  $d_{\text{C2}} = 1148$  
  $d_{\text{C3}} = 54$  
  $d_{\text{C4}} = 833$  
  $d_{\text{C5}} = 154$  
  $d_{\text{C6}} = 551$  
  $d_{\text{C7}} = 7081$  
  $d_{\text{C8}} = 76$  
  $d_{\text{C9}} = 66$  
  $d_{\text{C10}} = 174$  
  $d_{\text{C11}} = 15$  
  $d_{\text{C12}} = 680$  

- **Supply Capacities** (from `supply_capacity.csv`; column: `Unnamed: 0`, `supply_capacity`):  
  $s_{\text{S1}} = 4$  
  $s_{\text{S2}} = 575$  
  $s_{\text{S3}} = 1504$  
  $s_{\text{S4}} = 178$  
  $s_{\text{S5}} = 228$  
  $s_{\text{S6}} = 50$  
  $s_{\text{S7}} = 3$  
  $s_{\text{S8}} = 6148$  
  $s_{\text{S9}} = 6$  
  $s_{\text{S10}} = 10673$  
  $s_{\text{S11}} = 174$  

- **Transportation Costs** (from `transportation_costs.csv`; rows: `Unnamed: 0` = store, columns: customer group):  
  $c_{ij}$ is the entry in row $i$ (store) and column $j$ (customer group).

---

**All indices, coefficients, and constraints are mapped directly from the source columns and identifiers.**