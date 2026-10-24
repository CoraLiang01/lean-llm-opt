**Mathematical Model**

Let  

* \(K=\{C1,\dots ,C111\}\) components  
* \(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}\) workshops  

Decision variables  
\[
x_k \;\ge 0 \qquad \forall k\in K
\]

Parameters  

* \(t_{w,k}\) – processing time (hours) of component \(k\) in workshop \(w\)  
* \(p_k\) – unit price of component \(k\)  
* \(H_w\) – total available working hours in workshop \(w\)

\[
\begin{aligned}
\max\;&\sum_{k\in K} p_k\,x_k \\[4pt]
\text{s.t.}\;&\sum_{k\in K} t_{w,k}\,x_k \;\le\; H_w \qquad \forall w\in W \\[4pt]
&x_k \;\ge\; 0 \qquad \forall k\in K
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Row Identifier | Column Identifier |
|-----------|--------------|----------------|-------------------|
| \(t_{w,k}\) | `file_0_view_0` | `workshop` (row) | `C1`–`C111` (column) |
| \(p_k\) | `file_1_view_0` | `Unnamed: 0` (component id) | `unit_price` |
| \(H_w\) | `file_2_view_0` | `workshop` | `total_hours` |

All numeric values are taken directly from the corresponding cells in the tables; no additional processing or enumeration is required.