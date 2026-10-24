**Mathematical Model**

Let  

* \(I\) – set of drug products (index \(i\)).  
* \(v_i\) – benefit per unit of product \(i\).  
* \(w_i\) – weight (stock space) per unit of product \(i\).  
* \(C\) – total available stock capacity.  

Decision variable  

\[
x_i \ge 0 \quad\text{(units of product \(i\) ordered each day)}
\]

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(C\) | `file_0_view_0` | `Capacity` column (single value) |
| \(v_i\) | `file_1_view_0` | `Value` column, row‑to‑index mapping: index \(i\) corresponds to the row with `data_row_number` \(i\) |
| \(w_i\) | `file_1_view_0` | `Weight` column, same row‑to‑index mapping as above |

*Index set \(I\)* is the set of all rows returned from `file_1_view_0` (20 products).  
All parameters are taken directly from the CSV files; no additional limits or integer restrictions are imposed beyond non‑negativity.