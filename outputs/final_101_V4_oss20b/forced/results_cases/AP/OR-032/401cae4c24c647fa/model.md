**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product_Name* starts with the prefix **“Books”** (obtained from the filtered view `file_0_view_0`).  
- For each \(i\in I\) define the parameters  

\[
\begin{aligned}
r_i &\;=\; \text{Revenue of product } i \quad (\text{column } \texttt{Revenue})\\
d_i &\;=\; \text{Demand of product } i \quad (\text{column } \texttt{Demand})\\
s_i &\;=\; \text{Initial Inventory of product } i \quad (\text{column } \texttt{Initial Inventory})
\end{aligned}
\]

Decision variable  

\[
x_i \;\ge\; 0 \qquad \text{(units of product } i \text{ to be sold)}
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
x_i &\;\le\; d_i &&\forall i\in I \quad \text{(cannot sell more than demand)}\\[4pt]
x_i &\;\le\; s_i &&\forall i\in I \quad \text{(cannot sell more than available inventory)}\\[4pt]
x_i &\;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Index Column | Parameter Mapping |
|----------|--------------|-------------------|
| `file_0_view_0` | `Product_Name` | `r_i` ← `Revenue` |
| | | `d_i` ← `Demand` |
| | | `s_i` ← `Initial Inventory` |

The index set \(I\) is implicitly defined by the filter condition  
`Product_Name` starts with `"Books"` applied to the view `file_0_view_0`.