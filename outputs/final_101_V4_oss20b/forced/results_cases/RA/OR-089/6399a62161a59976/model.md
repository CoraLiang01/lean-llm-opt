**Mathematical Model**

Let  

* \(I=\{C1,\dots ,C15\}\) – set of customers (from *file_0_view_0* “Customer” column)  
* \(J=\{SC1,\dots ,SC10\}\) – set of service‑centres (from *file_1_view_0* “Service Center” column)

Parameters  

* \(f_j\) – fixed opening cost of centre \(j\)  
* \(c_{ij}\) – cost of serving customer \(i\) from centre \(j\)

Decision variables  

* \(y_j \in \{0,1\}\) – 1 if centre \(j\) is opened, 0 otherwise  
* \(x_{ij} \in \{0,1\}\) – 1 if customer \(i\) is assigned to centre \(j\), 0 otherwise  

---

\[
\begin{aligned}
\min\;& \sum_{j\in J} f_j\,y_j \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\
\text{s.t. } & \sum_{j\in J} x_{ij}=1, && \forall i\in I \qquad\text{(each customer served once)}\\[4pt]
& x_{ij}\le y_j, && \forall i\in I,\; j\in J \qquad\text{(assignment only to open centre)}\\[4pt]
& \sum_{i\in I} x_{ij}\le 4\,y_j, && \forall j\in J \qquad\text{(max 4 customers per open centre)}\\[4pt]
& x_{ij}\in\{0,1\}, && \forall i\in I,\; j\in J\\
& y_j\in\{0,1\}, && \forall j\in J
\end{aligned}
\]

---

### Data Mapping

| Parameter | Table ID | Column | Index Key |
|-----------|----------|--------|-----------|
| \(f_j\) | `file_1_view_0` | `Fixed Opening Cost` | `Service Center` |
| \(c_{ij}\) | `file_0_view_0` | `SC1` … `SC10` | `Customer` |

*The index keys (`Service Center` and `Customer`) provide the mapping from the parameter values to the corresponding elements of the sets \(J\) and \(I\). No other data are required for the model.*