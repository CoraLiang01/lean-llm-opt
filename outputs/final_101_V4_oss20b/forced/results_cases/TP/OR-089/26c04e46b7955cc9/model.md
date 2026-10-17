**Mathematical model**

Let  

* \(I=\{SC1,SC2,\dots ,SC10\}\) – set of candidate service centres  
* \(J=\{C1,C2,\dots ,C15\}\) – set of customers  

Decision variables  

\[
\begin{aligned}
y_i &\in \{0,1\} &&\text{(centre \(i\) is opened)}\\
x_{ij} &\in \{0,1\} &&\text{(customer \(j\) is served by centre \(i\))}
\end{aligned}
\]

Parameters  

\[
\begin{aligned}
f_i &\ge 0 &&\text{fixed opening cost of centre }i\\
c_{ij} &\ge 0 &&\text{service cost of assigning customer }j\text{ to centre }i
\end{aligned}
\]

Objective  

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Assignment** – each customer is served exactly once  
   \[
   \sum_{i\in I} x_{ij}=1 \qquad \forall j\in J
   \]

2. **Open‑only assignment** – a customer can be assigned only to an opened centre  
   \[
   x_{ij}\le y_i \qquad \forall i\in I,\; j\in J
   \]

3. **Capacity** – an opened centre serves at most 4 customers  
   \[
   \sum_{j\in J} x_{ij}\le 4 \qquad \forall i\in I
   \]

4. **Variable domains**  
   \[
   y_i\in\{0,1\},\; x_{ij}\in\{0,1\}\qquad \forall i\in I,\; j\in J
   \]

---

**Data‑mapping rules**

| Parameter / Variable | Source table | Key column(s) | Value column(s) | Notes |
|----------------------|--------------|---------------|-----------------|-------|
| \(I\) (service centres) | `file_1_view_0` | `Service Center` | – | 10 centres |
| \(J\) (customers) | `file_0_view_0` | `Customer` | – | 15 customers |
| \(f_i\) | `file_1_view_0` | `Service Center` | `Fixed Opening Cost` | Fixed cost per centre |
| \(c_{ij}\) | `file_0_view_0` | `Customer` (row) + `SC1`–`SC10` (columns) | `SC1`–`SC10` | Service cost matrix |
| \(y_i\) | – | – | – | Binary decision variable |
| \(x_{ij}\) | – | – | – | Binary decision variable |

All numerical values are taken verbatim from the CSV files; no aggregation or
transformation is performed. The model uses only the entities and parameters
explicitly defined above.