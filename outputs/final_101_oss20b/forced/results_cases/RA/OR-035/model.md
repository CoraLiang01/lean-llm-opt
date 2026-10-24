**Mathematical Model**

Let  

- \(I\) be the set of bread types indexed by the values of **ProductName** in *file_1_view_0*.  
- \(x_i\) – number of units of bread type \(i\) to order (integer, \(\ge 0\)).  

Parameters  

- \(\text{Value}_i\) – expected profit per unit of bread \(i\).  
- \(\text{Weight}_i\) – space requirement (weight) per unit of bread \(i\).  
- \(\text{Capacity}\) – total storage capacity.

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{Value}_i \, x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} \text{Weight}_i \, x_i \;\le\; \text{Capacity} \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(\text{Value}_i\) | file_1_view_0 | Value | For each record, set \(\text{Value}_{\text{ProductName}} = \text{Value}\). |
| \(\text{Weight}_i\) | file_1_view_0 | Weight | For each record, set \(\text{Weight}_{\text{ProductName}} = \text{Weight}\). |
| \(\text{Capacity}\) | file_0_view_0 | Capacity | Set \(\text{Capacity} = \text{Capacity}\) (single value). |

The index set \(I\) is implicitly defined as the set of all **ProductName** values present in *file_1_view_0*.