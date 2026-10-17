## Mathematical Model

Let  

* \(I=\{1,\dots,5\}\) – set of suppliers (Supplier1…Supplier5)  
* \(J=\{1,\dots,6\}\) – set of customers (Customer1…Customer6)

Parameters (all taken directly from the supplied CSV files)

| Parameter | Definition | Source |
|-----------|------------|--------|
| \(d_j\) | Daily demand of customer \(j\) | `customer_demand.csv`, column **demand** |
| \(s_i\) | Supply capacity of supplier \(i\) | `supply_capacity.csv`, column **supply_capacity** |
| \(c_{ij}\) | Transportation cost per unit from supplier \(i\) to customer \(j\) | `transportation_costs.csv`, matrix entry \((i,j)\) |

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\) (continuous, units of produce)

### Objective
\[
\min Z = \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

### Constraints  

1. **Supplier capacity**  
\[
\sum_{j\in J} x_{ij} \;\le\; s_i , \qquad \forall i\in I
\]

2. **Customer demand**  
\[
\sum_{i\in I} x_{ij} \;\ge\; d_j , \qquad \forall j\in J
\]

3. **Non‑negativity**  
\[
x_{ij}\;\ge\;0 , \qquad \forall i\in I,\; j\in J
\]

---

## Data Mapping

| Index | Table ID | Column(s) | Mapping |
|-------|----------|-----------|---------|
| \(i\) | `file_1_view_0` | `Suppliers` | Supplier identifiers (Supplier1…Supplier5) |
| \(j\) | `file_0_view_0` | `Customers` | Customer identifiers (Customer1…Customer6) |
| \(s_i\) | `file_1_view_0` | `supply_capacity` | Supplier capacity |
| \(d_j\) | `file_0_view_0` | `demand` | Customer demand |
| \(c_{ij}\) | `file_2_view_0` | matrix entry (row `Suppliers`, column `Customers`) | Transportation cost |

All parameters are taken as numeric values from the corresponding CSV files; no additional processing or enumeration of rows is required. The model uses continuous shipment quantities and respects the capacity and demand constraints exactly as specified.