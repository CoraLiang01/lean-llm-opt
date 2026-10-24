**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & Z = \sum_{i\in I} r_i\,x_i \\[4pt]
\text{subject to}\quad
& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le s_i &&\forall i\in I \\[4pt]
& x_i \ge 0 &&\forall i\in I
\end{aligned}
\]

**Index set**

\[
I = \{\, i \mid \text{Product Name of row } i \text{ contains the substring “Baby”}\,\}
\]

**Parameters**

\[
\begin{aligned}
r_i &\;=\; \text{Revenue of product } i \\
d_i &\;=\; \text{Demand of product } i \\
s_i &\;=\; \text{Initial Inventory of product } i
\end{aligned}
\]

**Decision variables**

\[
x_i \;\ge\; 0 \quad \text{(units of product } i \text{ to fulfill)}
\]

---

### Data Mapping Rules

| Table ID | Source File | Columns | Index | Parameters |
|----------|-------------|---------|-------|------------|
| file_0_view_0 | EuropeSalesRecords.csv | Product Name, Revenue, Demand, Initial Inventory | `i = Product Name` | `r_i = Revenue`, `d_i = Demand`, `s_i = Initial Inventory` |

*Filter applied:* Only rows where **Product Name** contains the substring `"Baby"` are included in the index set \(I\).