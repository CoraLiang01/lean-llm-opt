**Mathematical Model**

Let  

- \(I=\{S1,\dots ,S12\}\) (Distribution centers)  
- \(J=\{C1,\dots ,C12\}\) (Customer groups)

**Parameters**

- \(d_j\) – daily demand of customer \(j\)  \(j\in J\)  (“customer_demand.csv”, column *demand*)  
- \(s_i\) – supply capacity of center \(i\)  \(i\in I\)  (“supply_capacity.csv”, column *supply_capacity*)  
- \(c_{ij}\) – transportation cost per unit from center \(i\) to customer \(j\) \(i\in I,\;j\in J\) (“transportation_costs.csv”, matrix)

**Decision Variables**

- \(x_{ij}\ge 0\) – quantity shipped from center \(i\) to customer \(j\) \(i\in I,\;j\in J\)

**Objective**

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j,\qquad \forall j\in J
   \]

2. **Supply capacity**  
   \[
   \sum_{j\in J} x_{ij}\le s_i,\qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\ge 0,\qquad \forall i\in I,\;j\in J
   \]

---

**Data Mapping**

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| \(d_j\) | `file_0_view_0` | `demand` (row key: `customer`) |
| \(s_i\) | `file_1_view_0` | `supply_capacity` (row key: `Unnamed: 0`) |
| \(c_{ij}\) | `file_2_view_0` | matrix: rows `Unnamed: 0` (suppliers), columns `C1`–`C12` (customers) |

All indices \(i\) and \(j\) are taken from the row and column identifiers of the respective tables. No additional constraints or parameters are introduced.