**Mathematical Model (FLP)**  

Let  

* \(I=\{S1,\dots ,S11\}\) – set of Walmart stores (suppliers)  
* \(J=\{C1,\dots ,C12\}\) – set of customer groups  

**Parameters**  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(d_j\) | Daily demand of customer \(j\) | `file_0_view_0` column **demand** (row id **customer**) |
| \(s_i\) | Daily supply capacity of store \(i\) | `file_1_view_0` column **supply_capacity** (row id **Unnamed: 0**) |
| \(c_{ij}\) | Transportation cost per unit from store \(i\) to customer \(j\) | `file_2_view_0` matrix (row id **Unnamed: 0**, column id **customer**) |

**Decision Variables**  

* \(x_{ij}\ge 0\) – quantity shipped from store \(i\) to customer \(j\) (continuous)

**Objective**  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
   \]

2. **Supply capacity**  
   \[
   \sum_{j\in J} x_{ij}\le s_i ,\qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\ge 0 ,\qquad \forall i\in I,\; j\in J
   \]

---

### Data Mapping Rules

| Parameter | Table ID | Row ID column | Value column | Notes |
|-----------|----------|---------------|--------------|-------|
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | Each row gives demand for one customer |
| \(s_i\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | Each row gives capacity for one store |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` (row) | `C1`–`C12` (columns) | Matrix of unit costs; row id maps to store, column id maps to customer |

These mappings ensure that every parameter value used in the model is directly traceable to the corresponding cell in the source CSV files. No additional constraints or variables are introduced beyond those required by the problem statement.