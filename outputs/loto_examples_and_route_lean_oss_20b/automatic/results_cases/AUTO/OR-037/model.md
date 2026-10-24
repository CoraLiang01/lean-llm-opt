**Mathematical Model**

Let  

- \(I=\{1,\dots,25\}\) be the index set of vehicle types (the 25 rows of *products.csv*).  
- \(x_i\in\mathbb{Z}_{+}\) be the number of vehicles of type \(i\) ordered per day.  

Parameters  

- \(v_i\) – profit value of type \(i\) (column *Value* in *products.csv*).  
- \(w_i\) – weight (inventory space) of type \(i\) (column *Weight* in *products.csv*).  
- \(C\) – total inventory capacity (column *Capacity* in *capacity.csv*).  

\[
\begin{aligned}
\max\;&\sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;&\sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
&x_i\;\in\;\mathbb{Z}_{+}\qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column | Row(s) |
|-----------|--------|----------|--------|--------|
| \(C\) | capacity.csv | `file_0_view_0` | `Capacity` | row 0 |
| \(v_i\) | products.csv | `file_1_view_0` | `Value` | row i (0‑based) |
| \(w_i\) | products.csv | `file_1_view_0` | `Weight` | row i (0‑based) |
| Index set \(I\) | products.csv | `file_1_view_0` | `ProductName` | all 25 rows |

All values are taken directly from the corresponding CSV files; no additional data or assumptions are introduced.