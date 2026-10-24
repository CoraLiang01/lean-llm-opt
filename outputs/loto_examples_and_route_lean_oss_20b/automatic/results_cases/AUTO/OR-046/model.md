**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) be the set of products (index by the order of the rows in *products.csv*).  
* \(x_i\) – number of units of product \(i\) ordered each day.  

Parameters (obtained from the source tables)

| Parameter | Source | Description |
|-----------|--------|-------------|
| \(C\) | *capacity.csv*, column **Capacity** | Total stock‑capacity (875) |
| \(w_i\) | *products.csv*, column **Weight** | Weight of one unit of product \(i\) |
| \(v_i\) | *products.csv*, column **Value** | Profit (benefit) per unit of product \(i\) |

---

### Objective
\[
\max \;\; \sum_{i\in I} v_i\,x_i
\]

### Constraints
\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

\[
x_i \;\ge\; 0,\qquad x_i \in \mathbb{Z}\;\;\forall i\in I
\]

---

**Data Mapping**

* **Table** `file_0_view_0`  
  * Column `Capacity` → parameter \(C\)

* **Table** `file_1_view_0`  
  * Columns `Weight` → parameters \(w_i\) (indexed by row order)  
  * Columns `Value` → parameters \(v_i\) (indexed by row order)

All products returned in *products.csv* (10 rows) form the index set \(I\). No additional filtering or aggregation is applied.