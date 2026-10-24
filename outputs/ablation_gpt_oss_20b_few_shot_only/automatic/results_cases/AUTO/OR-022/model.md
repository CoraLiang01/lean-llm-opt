**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* contains the string “27in”.  
- For each \(i\in I\)  

\[
\begin{aligned}
r_i &\;=\; \text{Revenue of product }i \\
d_i &\;=\; \text{Demand of product }i \\
I_i &\;=\; \text{Initial Inventory of product }i \\
x_i &\;=\; \text{Number of units of product }i\text{ to be fulfilled}
\end{aligned}
\]

---

### Objective
\[
\max \;\sum_{i\in I} r_i\,x_i
\]

### Constraints
\[
\begin{aligned}
&0 \;\le\; x_i \;\le\; d_i &&\forall i\in I \quad\text{(Demand limit)}\\[4pt]
&0 \;\le\; x_i \;\le\; I_i &&\forall i\in I \quad\text{(Inventory limit)}\\[4pt]
&x_i \in \mathbb{Z} &&\forall i\in I \quad\text{(Integer decision)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Source Column |
|-----------|-------------|---------------|
| \(r_i\) (Revenue) | `Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv` | `Revenue` |
| \(d_i\) (Demand) | `Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv` | `Demand` |
| \(I_i\) (Initial Inventory) | `Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv` | `Initial Inventory` |
| Index set \(I\) | Same file | `Product Name` (rows containing “27in”) |

No additional capacity or trading‑eligibility constraints are imposed, as they are not specified in the current query.