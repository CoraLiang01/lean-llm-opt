**Mathematical Model**

Sets  
\[
I \;=\; \{\,i \mid i \text{ is a product name in } \texttt{file\_1\_view\_0}\,\}
\]

Parameters  
\[
\begin{aligned}
v_i &\;=\; \text{Value of product } i \quad (\texttt{file\_1\_view\_0.Value})\\
w_i &\;=\; \text{Weight (space) of product } i \quad (\texttt{file\_1\_view\_0.Weight})\\
C   &\;=\; \text{Total storage capacity} \quad (\texttt{file\_0\_view\_0.Capacity})
\end{aligned}
\]

Decision Variables  
\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\quad \forall i\in I
\]

Objective  
\[
\max \;\sum_{i\in I} v_i\,x_i
\]

Capacity Constraint  
\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

---

**Data Mapping**

| Parameter | Table ID          | Column   |
|-----------|-------------------|----------|
| \(v_i\)   | `file_1_view_0`   | `Value`  |
| \(w_i\)   | `file_1_view_0`   | `Weight` |
| \(C\)     | `file_0_view_0`   | `Capacity` |
| \(I\)     | `file_1_view_0`   | `ProductName` |

All parameters are taken directly from the specified tables; no additional data are introduced.