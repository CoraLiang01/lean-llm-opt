**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* contains the string “27in”.  
- For each \(i\in I\):

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(R_i\) | Revenue per unit | `Revenue` column of table `file_0_view_0` |
| \(D_i\) | Total demand (deterministic) | `Demand` column of table `file_0_view_0` |
| \(I_i\) | Initial inventory | `Initial Inventory` column of table `file_0_view_0` |

Decision variable  

- \(x_i\) – number of units of product \(i\) to be fulfilled.  
  \(x_i \in \mathbb{Z}_{+}\) (positive integer).

The model is

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\
\text{s.t. } & x_i \;\le\; D_i && \forall i\in I \quad\text{(demand limit)}\\
& x_i \;\le\; I_i && \forall i\in I \quad\text{(inventory limit)}\\
& x_i \;\ge\; 1 && \forall i\in I \quad\text{(positive integer)}\\
& x_i \in \mathbb{Z} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(R_i\) | `file_0_view_0` | `Revenue` |
| \(D_i\) | `file_0_view_0` | `Demand` |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is derived from the same table by selecting rows where *Product Name* contains “27in”. No additional data or constraints are introduced beyond those specified in the query.