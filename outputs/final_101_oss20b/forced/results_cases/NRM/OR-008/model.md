**Mathematical Model**

\[
\begin{aligned}
\text{Minimize}\quad & Z = \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{subject to}\quad
& \sum_{i\in I} x_{ij} = d_j , &&\forall j\in J \quad\text{(demand)}\\
& \sum_{j\in J} x_{ij} \le s_i , &&\forall i\in I \quad\text{(supply)}\\
& x_{ij}\;\ge\;0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

**Index Sets**

- \(I\) – set of suppliers (warehouses)  
- \(J\) – set of customers (stores)

**Parameters**

- \(d_j\) – demand of customer \(j\)  
- \(s_i\) – supply capacity of supplier \(i\)  
- \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\)

**Decision Variables**

- \(x_{ij}\) – amount of produce shipped from supplier \(i\) to customer \(j\) (continuous, \(\ge 0\))

---

### Data Mapping

| Parameter / Set | Source Table | Source Column(s) | Notes |
|-----------------|--------------|------------------|-------|
| \(J\) | `file_0_view_0` | `Customers` | Index set of customers |
| \(d_j\) | `file_0_view_0` | `demand` | Demand per customer |
| \(I\) | `file_1_view_0` | `Suppliers` | Index set of suppliers |
| \(s_i\) | `file_1_view_0` | `supply_capacity` | Supply capacity per supplier |
| \(c_{ij}\) | `file_2_view_0` | Matrix cells (`Unnamed: 0` rows, `Customer1`–`Customer6` columns) | Cost per unit from supplier to customer |

The matrix `file_2_view_0` is validated to have shape \([5,6]\) with row IDs aligned to `Suppliers` and column IDs aligned to `Customers`. No numeric values are listed here; the model references the parameters symbolically.