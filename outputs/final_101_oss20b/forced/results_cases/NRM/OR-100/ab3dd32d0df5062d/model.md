##### Index Sets  

\[
I=\{1,2,\dots ,111\}
\]

\[
W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}
\]

##### Parameters  

\[
t_{i,w}\quad\text{(processing time of component }i\text{ in workshop }w\text{)} 
\]

\[
p_i\quad\text{(unit price of component }i\text{)}
\]

\[
H_w\quad\text{(total available working hours in workshop }w\text{)}
\]

##### Decision Variables  

\[
x_i\in \mathbb{Z}_{\ge 0}\quad\forall i\in I
\]

##### Objective Function  

\[
\max \;\sum_{i\in I} p_i\,x_i
\]

##### Constraints  

###### Workshop capacity constraints  

\[
\sum_{i\in I} t_{i,w}\,x_i \;\le\; H_w \quad\forall w\in W
\]

###### Non‑negativity and integrality  

\[
x_i \in \mathbb{Z}_{\ge 0}\quad\forall i\in I
\]

##### Data Mapping  

| Parameter | Source table_id | Source column(s) |
|-----------|-----------------|------------------|
| \(t_{i,w}\) | file_0_view_0 | Columns C1 … C111 (each column corresponds to a component; rows correspond to workshops: Casting, Milling, Finishing, Assembly, QA & Packaging) |
| \(p_i\) | file_1_view_0 | Column `unit_price` (row identified by `Unnamed: 0` = C1 … C111) |
| \(H_w\) | file_2_view_0 | Column `total_hours` (row identified by `workshop` = Casting, Milling, Finishing, Assembly, QA & Packaging) |

All parameter values are taken directly from the retrieved CSV files; no additional data or assumptions are introduced.