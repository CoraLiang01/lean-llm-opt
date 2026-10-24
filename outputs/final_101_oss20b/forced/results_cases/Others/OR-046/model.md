**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) be the set of products.  
* \(x_i\) – number of units of product \(i\) ordered each day.  

Parameters (obtained from the supplied tables)

| Parameter | Source | Description |
|-----------|--------|-------------|
| \(w_i\) | `products.csv` – column **Weight** | Weight (kg) of one unit of product \(i\) |
| \(v_i\) | `products.csv` – column **Value** | Monetary benefit (yuan) per unit of product \(i\) |
| \(C\) | `capacity.csv` – column **Capacity** | Total available stock capacity (kg) |

The model is

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0,\;\; x_i \in \mathbb{Z}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Index | Table ID | Column | Mapping |
|-------|----------|--------|---------|
| \(i\) | `file_1_view_0` | `ProductName` | Product identifier |
| \(w_i\) | `file_1_view_0` | `Weight` | Weight of product \(i\) |
| \(v_i\) | `file_1_view_0` | `Value` | Value of product \(i\) |
| \(C\) | `file_0_view_0` | `Capacity` | Total capacity |

All parameters are taken directly from the corresponding columns of the returned rows; no additional filtering or aggregation is applied. The decision variables \(x_i\) are integer and non‑negative, reflecting whole units of each product.