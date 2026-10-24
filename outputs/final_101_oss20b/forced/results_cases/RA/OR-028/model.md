**Mathematical Model**

Let  

- \(I\) be the set of all products (indexed by the value in the *Product Name* column).  
- For each \(i\in I\) let  

  \[
  \begin{aligned}
  r_i &\;=\;\text{Revenue per unit of product }i,\\
  d_i &\;=\;\text{Demand for product }i,\\
  s_i &\;=\;\text{Initial inventory of product }i.
  \end{aligned}
  \]

- Decision variable  

  \[
  x_i \;=\;\text{number of units of product }i\text{ that will be fulfilled}.
  \]

---

### Objective
\[
\max \;\sum_{i\in I} r_i\,x_i
\]

### Constraints
\[
\begin{aligned}
&0 \;\le\; x_i \;\le\; d_i &&\forall i\in I \quad\text{(cannot exceed demand)}\\[4pt]
&0 \;\le\; x_i \;\le\; s_i &&\forall i\in I \quad\text{(cannot exceed inventory)}\\[4pt]
&x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I \quad\text{(integer units)}
\end{aligned}
\]

---

## Data Mapping

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(r_i\) (Revenue) | `file_0_view_0` | `Revenue` |
| \(d_i\) (Demand) | `file_0_view_0` | `Demand` |
| \(s_i\) (Initial Inventory) | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is derived from the distinct values in the *Product Name* column of the same table. No other data are required.