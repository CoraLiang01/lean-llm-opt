##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$ (continuous).

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses, from supply_capacity.csv)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores, from customer_demand.csv)

##### Parameters

- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each warehouse $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $d_j$ (store demand):  
  From `customer_demand.csv` (`file_0_view_0`), columns:  
  - $d_{\text{D1}} = 428$  
  - $d_{\text{D2}} = 217$  
  - $d_{\text{D3}} = 214$  
  - $d_{\text{D4}} = 380$  
  - $d_{\text{D5}} = 254$  

- $s_i$ (warehouse supply capacity):  
  From `supply_capacity.csv` (`file_1_view_0`), columns:  
  - $s_{\text{S1}} = 428$  
  - $s_{\text{S2}} = 217$  
  - $s_{\text{S3}} = 214$  
  - $s_{\text{S4}} = 380$  
  - $s_{\text{S5}} = 254$  

- $c_{ij}$ (transportation cost):  
  From `transportation_costs.csv` (`file_2_view_0`),  
  - Rows: warehouses $i$ (`Unnamed: 0`): S1, S2, S3, S4, S5  
  - Columns: stores $j$: D1, D2, D3, D4, D5  
  - $c_{ij}$ is the value in row $i$, column $j$.

##### Table References

- $d_j$: `file_0_view_0`, column `demand`, indexed by `customer`
- $s_i$: `file_1_view_0`, column `supply_capacity`, indexed by `region`
- $c_{ij}$: `file_2_view_0`, row `Unnamed: 0` (warehouse), columns D1–D5 (store)

---

**All parameters and indices are bound exactly to the retrieved data.**