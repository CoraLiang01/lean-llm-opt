**Mathematical Model**

Let  

\[
P=\{P1,P2,P3\},\qquad R=\{R1,R2,R3,R4,R5,R6\}
\]

Parameters  

\[
\begin{aligned}
s_i &\;=\;\text{SupplyCapacity of plant }i && (i\in P)\\
d_j &\;=\;\text{Demand of retailer }j && (j\in R)\\
c_{ij}&=\;\text{Variable cost per carton on route }i\!-\!j && (i\in P,\;j\in R)\\
f_{ij}&=\;\text{Fixed activation cost of route }i\!-\!j && (i\in P,\;j\in R)\\
M_{ij}&=\min\{s_i,d_j\} && (i\in P,\;j\in R)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
x_{ij}&\ge 0 &&\text{cartons shipped on route }i\!-\!j\\
y_{ij}&\in\{0,1\} &&\text{route }i\!-\!j\text{ is opened}
\end{aligned}
\]

Objective  

\[
\min \;\sum_{i\in P}\sum_{j\in R}\bigl(c_{ij}\,x_{ij}+f_{ij}\,y_{ij}\bigr)
\]

Constraints  

1. **Retailer demand**  
\[
\sum_{i\in P}x_{ij}=d_j \qquad \forall j\in R
\]

2. **Plant supply**  
\[
\sum_{j\in R}x_{ij}\le s_i \qquad \forall i\in P
\]

3. **Activation linking**  
\[
x_{ij}\le M_{ij}\,y_{ij}\qquad \forall i\in P,\;j\in R
\]

4. **Variable bounds**  
\[
x_{ij}\ge 0,\qquad y_{ij}\in\{0,1\}\qquad \forall i\in P,\;j\in R
\]

---

### Data Mapping

| Parameter | Source | Table ID | Columns | Mapping |
|-----------|--------|----------|---------|---------|
| \(s_i\) | plant_capacity.csv | file_0_view_0 | Plant, SupplyCapacity | \(s_i =\) value of **SupplyCapacity** where **Plant** = \(i\) |
| \(d_j\) | retailer_demand.csv | file_1_view_0 | Retailer, Demand | \(d_j =\) value of **Demand** where **Retailer** = \(j\) |
| \(c_{ij}\) | route_variable_costs.csv | file_2_view_0 | Plant, R1–R6 | \(c_{ij} =\) value in column **Rj** where **Plant** = \(i\) |
| \(f_{ij}\) | route_fixed_costs.csv | file_3_view_0 | Plant, R1–R6 | \(f_{ij} =\) value in column **Rj** where **Plant** = \(i\) |
| \(M_{ij}\) | derived | – | – | \(M_{ij}=\min\{s_i,d_j\}\) |

Index sets \(P\) and \(R\) are taken from the **Plant** column of file_0_view_0 and the **Retailer** column of file_1_view_0, respectively. No other rows or columns are used.