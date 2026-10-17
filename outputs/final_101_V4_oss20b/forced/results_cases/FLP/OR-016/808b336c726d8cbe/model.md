**Decision Variables**

\[
x_i \ge 0 \qquad \forall i\in I
\]

*\(x_i\)* – quantity of product *i* sold (fulfilled).

---

**Parameters**

\[
\begin{aligned}
r_i &\;=\; \text{Revenue of product }i,\\
d_i &\;=\; \text{Demand of product }i,\\
s_i &\;=\; \text{Initial Inventory of product }i,
\end{aligned}
\qquad \forall i\in I
\]

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I &&\text{(cannot exceed demand)}\\[4pt]
x_i &\le s_i &&\forall i\in I &&\text{(cannot exceed inventory)}\\[4pt]
x_i &\ge 0 &&\forall i\in I
\end{aligned}
\]

---

### Data Mapping

| Index | Set | Parameter | Source |
|-------|-----|-----------|--------|
| \(I\) | \{Product Name\} | – | `file_0_view_0` (all rows) |
| \(r_i\) | – | Revenue of product *i* | `file_0_view_0` column **Revenue** |
| \(d_i\) | – | Demand of product *i* | `file_0_view_0` column **Demand** |
| \(s_i\) | – | Initial Inventory of product *i* | `file_0_view_0` column **Initial Inventory** |

*Mapping rule:* for each product name \(i\) in the table, assign the corresponding column values to \(r_i\), \(d_i\), and \(s_i\). No other data are required.