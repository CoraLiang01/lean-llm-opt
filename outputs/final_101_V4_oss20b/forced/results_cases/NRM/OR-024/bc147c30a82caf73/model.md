**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product Name* starts with **S700_** (12 items in the current observation).  

Parameters (indexed by \(i\in I\))  

\[
\begin{aligned}
r_i &\;=\;\text{Revenue of product }i \quad &(\text{from column ``Revenue''})\\
d_i &\;=\;\text{Demand of product }i \quad &(\text{from column ``Demand''})\\
q_i &\;=\;\text{Initial inventory of product }i \quad &(\text{from column ``Initial Inventory''})
\end{aligned}
\]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{+}\quad\text{(positive integer, number of units of product }i\text{ to sell)}
\]

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

1. **Demand limits**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limits**  
   \[
   x_i \;\le\; q_i \qquad \forall i\in I
   \]

3. **Non‑negativity / positivity**  
   \[
   x_i \;\ge\; 1 \qquad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(q_i\)   | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is implicitly defined by all rows in `file_0_view_0` whose *Product Name* begins with the prefix `S700_`. No additional data or constraints are introduced beyond those specified in the query.