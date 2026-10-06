##### Decision Variables

For each supplier $i \in I = \{\text{S1}, \text{S2}, \text{S3}\}$ and each customer $j \in J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$,

$$
x_{ij} \geq 0
$$

where $x_{ij}$ is the quantity shipped from supplier $i$ to customer $j$ (continuous).

---

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

where $c_{ij}$ is the unit transportation cost from supplier $i$ to customer $j$.

---

##### Constraints

1. **Demand satisfaction** (each customer receives at least its demand):

   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$

   where $d_j$ is the demand of customer $j$.

2. **Supply capacity** (each supplier does not exceed its capacity):

   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$

   where $s_i$ is the supply capacity of supplier $i$.

3. **Non-negativity**:

   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

##### Data Mapping

- **Sets:**
  - $I = \{\text{S1}, \text{S2}, \text{S3}\}$ (from `supply_capacity.csv`, `Unnamed: 0`, rows 0–2)
  - $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (from `customer_demand.csv`, `customer`, rows 0–3)

- **Parameters:**
  - $d_j$ (customer demand): from `customer_demand.csv`, column `demand`, for $j \in J$
    - $d_{\text{C1}} = 11$
    - $d_{\text{C2}} = 1148$
    - $d_{\text{C3}} = 54$
    - $d_{\text{C4}} = 833$
  - $s_i$ (supply capacity): from `supply_capacity.csv`, column `supply_capacity`, for $i \in I$
    - $s_{\text{S1}} = 4$
    - $s_{\text{S2}} = 575$
    - $s_{\text{S3}} = 1504$
  - $c_{ij}$ (transportation cost): from `transportation_costs.csv`, rows with `Unnamed: 0` in $I$, columns in $J$
    - $c_{\text{S1},\text{C1}} = 0.639144476970582$
    - $c_{\text{S1},\text{C2}} = 49.71842803015729$
    - $c_{\text{S1},\text{C3}} = 33.75857739960576$
    - $c_{\text{S1},\text{C4}} = 1570.673110465785$
    - $c_{\text{S2},\text{C1}} = 605.4786373569875$
    - $c_{\text{S2},\text{C2}} = 64.53562572761275$
    - $c_{\text{S2},\text{C3}} = 478.4779031378926$
    - $c_{\text{S2},\text{C4}} = 887.0480739088434$
    - $c_{\text{S3},\text{C1}} = 1139.0440074582496$
    - $c_{\text{S3},\text{C2}} = 4.785056325458736$
    - $c_{\text{S3},\text{C3}} = 1805.6214229758102$
    - $c_{\text{S3},\text{C4}} = 1302.8958147418275$

---

##### Table and Column Mapping

- $d_j$: `file_0_view_0` (customer_demand.csv), column `demand`, row_id_mapping $j$ (`customer`)
- $s_i$: `file_1_view_0` (supply_capacity.csv), column `supply_capacity`, row_id_mapping $i$ (`Unnamed: 0`)
- $c_{ij}$: `file_2_view_0` (transportation_costs.csv), row_id_mapping $i$ (`Unnamed: 0`), column_id_mapping $j$ (column names)

---

##### Complete Model

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
$$
$$
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
$$

with all parameters and indices as mapped above.