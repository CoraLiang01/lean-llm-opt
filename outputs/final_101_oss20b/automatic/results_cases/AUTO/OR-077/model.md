**Mathematical Model**

Let  

\[
I=\{i\mid i\text{ is an item in table }file\_0\_view\_0\}
\]

Parameters (obtained from the table)

\[
\begin{aligned}
v_i &:= \text{value of item }i \quad (\text{column }value)\\
w_i &:= \text{weight of item }i \quad (\text{column }weight)
\end{aligned}
\]

Decision variables  

\[
x_i \in \{0,1\}\quad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; 15
\]

---

**Data Mapping**

| Table | Column | Parameter |
|-------|--------|-----------|
| file_0_view_0 | item | index set \(I\) |
| file_0_view_0 | value | \(v_i\) |
| file_0_view_0 | weight | \(w_i\) |

No filtering is applied; all 140 items from the table are included in the index set. The capacity value 15 is a constant given in the problem statement.