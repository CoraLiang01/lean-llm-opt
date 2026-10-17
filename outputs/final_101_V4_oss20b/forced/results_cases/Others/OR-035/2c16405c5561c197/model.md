**Mathematical Model**

Let  

* \(I=\{1,\dots,10\}\) be the set of bread types.  
* \(x_i\in\mathbb{Z}_{\ge 0}\) – number of units of type \(i\) ordered each day.  

Parameters  

* \(\text{Profit}_i\) – expected profit per unit of type \(i\).  
* \(\text{Weight}_i\) – storage space required per unit of type \(i\).  
* \(\text{Capacity}\) – total available storage space (units).  

\[
\begin{aligned}
\max\;& \sum_{i\in I}\text{Profit}_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I}\text{Weight}_i\,x_i \;\le\; \text{Capacity} \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column(s) | Mapping to Model |
|----------|-----------|------------------|
| `file_0_view_0` | `Capacity` | \(\text{Capacity}\) (scalar) |
| `file_1_view_0` | `ProductName`, `Value`, `Weight` | For each row \(r\in I\):<br>• \(\text{Profit}_r \gets\) `Value`<br>• \(\text{Weight}_r \gets\) `Weight` |

*Index set \(I\)* is defined by the 10 rows returned from `file_1_view_0`.  
All parameters are taken directly from the corresponding columns; no additional processing or enumeration of numeric values is required.