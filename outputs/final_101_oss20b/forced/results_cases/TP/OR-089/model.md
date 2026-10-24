**Mathematical model**

Let  

* \(I=\{SC1,SC2,\dots ,SC10\}\) – set of service‑center candidates  
* \(J=\{C1,C2,\dots ,C15\}\) – set of customers  

Decision variables  

\[
y_i \in \{0,1\}\quad\forall i\in I
\]
\[
x_{ij}\in\{0,1\}\quad\forall i\in I,\;j\in J
\]

Parameters  

\[
f_i \ge 0 \quad\text{(fixed opening cost of centre }i)
\]
\[
c_{ij}\ge 0 \quad\text{(service cost of assigning customer }j\text{ to centre }i)
\]

Objective  

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Customer assignment** – each customer is served by exactly one centre  

\[
\sum_{i\in I} x_{ij}=1 \qquad \forall j\in J
\]

2. **Link to opening** – a customer can be assigned only to an opened centre  

\[
x_{ij}\le y_i \qquad \forall i\in I,\;j\in J
\]

3. **Capacity** – an opened centre serves at most 4 customers  

\[
\sum_{j\in J} x_{ij}\le 4 \qquad \forall i\in I
\]

4. **Variable domains** – already stated above.  

---

### Data‑mapping rules

| Parameter / Variable | Source table | Key column | Value column | Notes |
|----------------------|--------------|------------|--------------|-------|
| \(f_i\) | `file_1_view_0` | `Service Center` | `Fixed Opening Cost` | One row per centre |
| \(c_{ij}\) | `file_0_view_0` | `Customer` | `SC1`–`SC10` | One row per customer, one column per centre |
| \(y_i\) | – | – | – | Binary decision variable |
| \(x_{ij}\) | – | – | – | Binary decision variable |

*Index sets*  
- \(I\) is the set of values appearing in the `Service Center` column of `file_1_view_0`.  
- \(J\) is the set of values appearing in the `Customer` column of `file_0_view_0`.

All parameters are taken directly from the CSV files; no additional data are invented or omitted.