**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{npv}_i\,x_i \\[4pt]
\text{subject to}\quad
& \sum_{i\in I} \text{capital}_i\,x_i \;\le\; 1000 \\[4pt]
& x_4 + x_7 \;\le\; 1 \\[4pt]
& x_6 \;\le\; x_1 \\[4pt]
& x_{10} \;\le\; x_5 \\[4pt]
& x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

**Index set**

\[
I = \{1,2,\dots,110\}
\]

**Parameters**

\[
\begin{aligned}
\text{capital}_i &\ge 0 &&\text{(capital investment of project }i\text{ in k$)}\\
\text{npv}_i &\ge 0 &&\text{(expected NPV of project }i\text{ in k$)}
\end{aligned}
\]

**Decision variables**

\[
x_i \in \{0,1\}\quad \forall i\in I
\]

---

### Data Mapping Rules

| Table ID          | Column Name          | Parameter / Index |
|-------------------|----------------------|-------------------|
| `file_0_view_0`   | `Project ID`         | `i` (index)       |
| `file_0_view_0`   | `Capital (k$)`       | `capital_i`       |
| `file_0_view_0`   | `NPV (k$)`           | `npv_i`           |

*The column `Project Name` is not used in the model.*