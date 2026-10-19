##### Abstract Model

**Index Set**  
\(I\) – set of products with names beginning with “S700_”.

**Parameters**  
\(w_i\) – revenue per unit of product \(i\).  
\(d_i\) – deterministic demand for product \(i\).  
\(c_i\) – initial inventory of product \(i\).

**Decision Variables**  
\(x_i \in \mathbb{Z}\) – number of units of product \(i\) to fulfill.  
\(x_i \ge 1\) for all \(i \in I\).

**Objective Function**  
\[
\max \sum_{i \in I} w_i \, x_i
\]

**Constraints**

1. **Inventory Constraints**  
\[
x_i \le c_i , \quad \forall i \in I
\]

2. **Demand Constraints**  
\[
x_i \le d_i , \quad \forall i \in I
\]

3. **Variable Domain**  
\[
x_i \in \mathbb{Z}, \; x_i \ge 1 , \quad \forall i \in I
\]

---

##### Data Mapping

| Source | Table ID | Columns Used |
|--------|----------|--------------|
| SampleSalesData.csv | `file_0_view_0` | `Product Name`, `Revenue`, `Demand`, `Initial Inventory` |

The table `file_0_view_0` contains 12 rows that satisfy the filter `Product Name` starts with `S700_`. Each row provides the values for \(w_i\), \(d_i\), and \(c_i\) for the corresponding product \(i\).