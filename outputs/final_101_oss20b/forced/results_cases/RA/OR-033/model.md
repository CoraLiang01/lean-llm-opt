**Mathematical Model**

Sets  
\[
I \;=\; \{\text{baby products in the view } \texttt{file\_0\_view\_0}\}
\]

Parameters  
\[
\begin{aligned}
\text{Revenue}_i &\;=\; \text{Revenue of product } i \quad (\texttt{file\_0\_view\_0.Revenue})\\
\text{Demand}_i  &\;=\; \text{Demand of product } i \quad (\texttt{file\_0\_view\_0.Demand})\\
\text{InitialInventory}_i &\;=\; \text{Initial inventory of product } i \quad (\texttt{file\_0\_view\_0.Initial Inventory})
\end{aligned}
\]

Decision variables  
\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\quad \forall i\in I
\]

Objective  
\[
\max \;\sum_{i\in I} \text{Revenue}_i \, x_i
\]

Constraints  
\[
\begin{aligned}
x_i &\;\le\; \text{Demand}_i &&\forall i\in I\\
x_i &\;\le\; \text{InitialInventory}_i &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| Revenue_i | file_0_view_0 | Revenue |
| Demand_i  | file_0_view_0 | Demand |
| InitialInventory_i | file_0_view_0 | Initial Inventory |

The index set \(I\) is the set of product names returned by the filtered view `file_0_view_0`.