**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product Name* contains “4U”.  
- For each \(i\in I\) let  

  \[
  \begin{aligned}
  r_i &\;=\;\text{Revenue of product }i,\\
  d_i &\;=\;\text{Demand of product }i,\\
  \text{inv}_i &\;=\;\text{Initial Inventory of product }i.
  \end{aligned}
  \]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

represent the number of units of product \(i\) to be sold.

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; d_i &&\forall i\in I \quad\text{(cannot exceed demand)}\\[4pt]
x_i &\;\le\; \text{inv}_i &&\forall i\in I \quad\text{(cannot exceed inventory)}
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|---------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` |

The mapping uses the *Product Name* as the key to associate each parameter with its corresponding product. All numeric columns are interpreted as real numbers; the decision variables are constrained to be non‑negative integers.