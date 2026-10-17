**Symbolic Model**

\[
\begin{aligned}
\text{Sets} \quad & W \quad\text{: set of warehouses} \\
& P \quad\text{: set of product types} \\[4pt]

\text{Parameters} \quad & \text{Capacity}[w] \quad\forall w\in W \\
& \text{Value}[p] \quad\forall p\in P \\
& \text{Weight}[p] \quad\forall p\in P \quad(\text{not used in constraints}) \\[4pt]

\text{Decision Variables} \quad & x[p,w] \in \mathbb{Z}_{\ge 0}\quad\forall p\in P,\; w\in W \\[4pt]

\text{Objective} \quad & \max \;\sum_{p\in P}\;\text{Value}[p]\;\sum_{w\in W} x[p,w] \\[4pt]

\text{Constraints} \quad & \sum_{p\in P} x[p,w] \;\le\; \text{Capacity}[w] \qquad \forall w\in W \\[4pt]
& x[p,w] \;\ge\; 0 \qquad \forall p\in P,\; w\in W
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| Capacity[w] | `file_0_view_0` | `Capacity` |
| Value[p] | `file_1_view_0` | `Value` |
| Weight[p] | `file_1_view_0` | `Weight` |

*All indices \(w\) and \(p\) are taken from the rows of the corresponding source tables.*