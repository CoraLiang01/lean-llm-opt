**Mathematical Model**

Let  

* \(I\) – index set of all products whose name starts with “Aalop” (extracted from the table).  
* \(R_i\) – revenue per unit of product \(i\).  
* \(D_i\) – known deterministic demand for product \(i\).  
* \(I_i\) – initial inventory available for product \(i\).  
* \(x_i\) – number of units of product \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; D_i && \forall i\in I \quad\text{(demand limit)}\\
             & x_i \;\le\; I_i && \forall i\in I \quad\text{(inventory limit)}\\
             & x_i \;\ge\; 0 && \forall i\in I \\
             & x_i \in \mathbb{Z} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(R_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(D_i\) | Demand | `file_0_view_0` | `Demand` |
| \(I_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | Product names with prefix “Aalop” | `file_0_view_0` | `Product Name` |

The mapping uses the exact identifiers and column names from the retrieved CSV view. No numerical values are enumerated; the model remains symbolic and ready for instantiation with the actual data.