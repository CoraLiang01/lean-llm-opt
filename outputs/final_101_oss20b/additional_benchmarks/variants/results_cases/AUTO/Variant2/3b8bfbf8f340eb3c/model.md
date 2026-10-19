**Mathematical Model**

Let  

* \(I=\{S1,S2,S3,S4,S5,S6\}\) – set of suppliers (rows of *supplier_capacity.csv*)  
* \(J=\{C1,\dots ,C12\}\) – set of customers (rows of *customer_demand.csv*)

Parameters (all taken directly from the supplied tables)

| Parameter | Source table | Column(s) |
|-----------|--------------|-----------|
| \(s_i\) | *supplier_capacity.csv* | `SupplyCapacity` |
| \(d_j\) | *customer_demand.csv* | `Demand` |
| \(c_{ij}\) | *route_variable_costs.csv* | column `Cj` in row of supplier \(i\) |
| \(f_{ij}\) | *route_fixed_costs.csv* | column `Cj` in row of supplier \(i\) |

Decision variables  

\[
x_{ij}\ge 0 \quad\text{(continuous shipment quantity from }i\text{ to }j)
\]
\[
y_{ij}\in\{0,1\}\quad\text{(1 if route }i\!-\!j\text{ is used)}
\]

Let \(M\) be a sufficiently large constant, e.g.  
\(M=\sum_{j\in J} d_j\) (total demand) or \(M=\max_{i\in I}s_i\).

**Objective**

\[
\min \;\sum_{i\in I}\sum_{j\in J}\bigl(c_{ij}\,x_{ij}+f_{ij}\,y_{ij}\bigr)
\]

**Constraints**

1. **Customer demand satisfaction**  
   \[
   \sum_{i\in I}x_{ij}=d_j \qquad \forall j\in J
   \]

2. **Supplier capacity**  
   \[
   \sum_{j\in J}x_{ij}\le s_i \qquad \forall i\in I
   \]

3. **Link between shipment and route activation**  
   \[
   x_{ij}\le M\,y_{ij}\qquad \forall i\in I,\;j\in J
   \]

4. **Variable domains**  
   \[
   x_{ij}\ge 0,\qquad y_{ij}\in\{0,1\}\qquad \forall i\in I,\;j\in J
   \]

---

### Data Mapping Rules

| Index | Source | Column | Mapping |
|-------|--------|--------|---------|
| \(i\) | *supplier_capacity.csv* | `Supplier` | \(I\) |
| \(i\) | *route_variable_costs.csv* | `Supplier` | \(s_i\), \(c_{ij}\) |
| \(i\) | *route_fixed_costs.csv* | `Supplier` | \(f_{ij}\) |
| \(j\) | *customer_demand.csv* | `Customer` | \(J\) |
| \(j\) | *route_variable_costs.csv* | `Cj` | \(c_{ij}\) |
| \(j\) | *route_fixed_costs.csv* | `Cj` | \(f_{ij}\) |
| \(j\) | *customer_demand.csv* | `Demand` | \(d_j\) |

All parameters are taken verbatim from the corresponding table rows; no additional filtering or aggregation is applied. The model uses the full sets \(I\) and \(J\) as defined above. The big‑\(M\) constant is chosen to be at least the maximum possible shipment on any route, e.g. the total demand or the largest supplier capacity.