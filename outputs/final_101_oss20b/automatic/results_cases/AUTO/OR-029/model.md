**Symbolic Model**

Let  

- \(I\) be the set of FAUX products (identified by their *Product Name*).  
- For each \(i\in I\)  

\[
\begin{aligned}
R_i &\;=\; \text{Revenue of product } i,\\
D_i &\;=\; \text{Demand of product } i,\\
S_i &\;=\; \text{Initial Inventory of product } i.
\end{aligned}
\]

Decision variable  

\[
x_i \;\in\; \mathbb{Z}_{+}\qquad\text{(units of product }i\text{ to fulfill)}
\]

---

**Objective**

\[
\max \;\sum_{i\in I} R_i\,x_i
\]

---

**Constraints**

1. **Demand limits**  
   \[
   x_i \;\le\; D_i \qquad \forall i\in I
   \]

2. **Inventory limits**  
   \[
   x_i \;\le\; S_i \qquad \forall i\in I
   \]

3. **Non‑negativity** (implicit in \(\mathbb{Z}_{+}\))  

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | *Product Name* (filtered where contains “FAUX”) |
| \(R_i\) | `file_0_view_0` | `Revenue` |
| \(D_i\) | `file_0_view_0` | `Demand` |
| \(S_i\) | `file_0_view_0` | `Initial Inventory` |

All parameter values are taken directly from the rows of `file_0_view_0` that satisfy the filter condition `"Product Name" contains "FAUX"`. No additional data or constraints are introduced beyond those specified in the query.