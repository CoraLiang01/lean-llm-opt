**Mathematical Model**

Let  

* \(I=\{1,2\}\) be the set of 27‑inch products that satisfy the filter  
  “Product Name contains ‘27in’”.

Parameters (obtained from *Salesorders.csv* after filtering):

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(r_i\) | Revenue per unit of product \(i\) | `Revenue` column |
| \(d_i\) | Demand (units) of product \(i\) | `Demand` column |
| \(s_i\) | Initial inventory (units) of product \(i\) | `Initial Inventory` column |

Decision variables:

* \(x_i\) – number of units of product \(i\) to sell (integer, \(\ge 0\)).

Objective (maximize total revenue):

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

Subject to (cannot sell more than demand or available inventory):

\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I,\\
x_i &\le s_i &&\forall i\in I,\\
x_i &\in \mathbb{Z}_{\ge 0} &&\forall i\in I.
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Columns Used | Filter Condition | Index Set | Parameter Mapping |
|----------|--------------|------------------|-----------|-------------------|
| `file_0_view_0` | `Product Name`, `Revenue`, `Demand`, `Initial Inventory` | `Product Name` contains `"27in"` | \(I=\{1,2\}\) (row numbers 1 and 2 after filtering) | `Revenue` → \(r_i\) <br> `Demand` → \(d_i\) <br> `Initial Inventory` → \(s_i\) |

*The filter is applied before any indexing; only rows satisfying the condition are included in the index set.*