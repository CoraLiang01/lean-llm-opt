**Mathematical Model**

Let  

- \(I\) be the set of all products whose SKU starts with the prefix **ZZ** (the 5 records returned in the observation).  
- For each \(i\in I\) define  

\[
\begin{aligned}
R_i &\;=\;\text{Revenue of product }i,\\
D_i &\;=\;\text{Demand of product }i,\\
S_i &\;=\;\text{Initial inventory of product }i.
\end{aligned}
\]

Decision variable  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad\text{(units of product }i\text{ to fulfill)}.
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

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(R_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(D_i\) | Demand | `file_0_view_0` | `Demand` |
| \(S_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(i\) | SKU | `file_0_view_0` | `SKU` (prefix filter “ZZ”) |

All parameter values are taken directly from the rows returned in the observation; no additional data or assumptions are introduced.