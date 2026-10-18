##### Abstract Model

**Index Set**  
\(I\) – set of Aalop products.

**Parameters**  
\(w_i\) – revenue per unit of product \(i\).  
\(d_i\) – deterministic total demand for product \(i\) over the sales horizon.  
\(c_i\) – initial inventory available for product \(i\).

**Decision Variables**  
\(x_i \in \mathbb{Z}\) – number of units of product \(i\) the restaurant intends to fulfill.  
\(x_i \ge 1\) for all \(i \in I\).

**Objective Function**  
\[
\max \sum_{i \in I} w_i \, x_i
\]

**Constraints**

1. **Inventory Constraints**  
\[
x_i \le c_i, \quad \forall i \in I
\]

2. **Demand Constraints**  
\[
x_i \le d_i, \quad \forall i \in I
\]

3. **Variable Domain**  
\[
x_i \in \mathbb{Z}, \quad x_i \ge 1, \quad \forall i \in I
\]

---

##### Data Mapping

| Source | Table ID | Columns Used |
|--------|----------|--------------|
| Revenue, Demand, Initial Inventory, Product Name | `RestaurantSalesreport.csv` | `Revenue`, `Demand`, `Initial Inventory`, `Product Name` |

*The model above is symbolic; actual numerical values will be supplied from the specified CSV columns.*