##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses, from `supply_capacity.csv` and `transportation_costs.csv` row IDs)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores, from `customer_demand.csv` and `transportation_costs.csv` column IDs)

##### Parameters

- $d_j$: demand of store $j$ (from `customer_demand.csv`)
- $s_i$: supply capacity of warehouse $i$ (from `supply_capacity.csv`)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from `transportation_costs.csv`)

##### Data Mapping

- $d_j$ from `file_0_view_0` (`customer_demand.csv`):  
  - $d_{\text{D1}} = 428$
  - $d_{\text{D2}} = 217$
  - $d_{\text{D3}} = 214$
  - $d_{\text{D4}} = 380$
  - $d_{\text{D5}} = 254$

- $s_i$ from `file_1_view_0` (`supply_capacity.csv`):  
  - $s_{\text{S1}} = 428$
  - $s_{\text{S2}} = 217$
  - $s_{\text{S3}} = 214$
  - $s_{\text{S4}} = 380$
  - $s_{\text{S5}} = 254$

- $c_{ij}$ from `file_2_view_0` (`transportation_costs.csv`):  
  - $c_{ij}$ is the entry in row $i$ (warehouse, `Unnamed: 0`) and column $j$ (store, e.g., `D1`) of `file_2_view_0`.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]

2. **Supply capacity:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]

3. **Non-negativity:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

---

**Data Mapping Table**

| Symbol      | Source Table (table_id) | Column(s)         | Row(s)         | Mapping Description                                 |
|-------------|------------------------|-------------------|----------------|-----------------------------------------------------|
| $d_j$       | file_0_view_0          | customer, demand  | all            | $d_j$ is demand for store $j$                       |
| $s_i$       | file_1_view_0          | region, supply_capacity | all      | $s_i$ is supply capacity for warehouse $i$           |
| $c_{ij}$    | file_2_view_0          | Unnamed: 0, D1-D5 | all            | $c_{ij}$ is cost from warehouse $i$ to store $j$     |

---

**Complete Model**

\[
\begin{align*}
\min_{x_{ij} \geq 0} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{align*}
\]

with all parameters and index sets as mapped above.