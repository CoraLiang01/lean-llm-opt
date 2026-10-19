##### Index Sets  
- \(M=\{M_1,M_2,\dots ,M_{12}\}\) (12 machines)  
- \(T=\{A,B,C,D,E,F,G,H,I,J,K,L\}\) (12 tasks)

##### Parameters  
- \(c_{ij}\) = machining cost incurred when machine \(i\in M\) is assigned to task \(j\in T\)

##### Decision Variables  
- \(x_{ij}\in\{0,1\}\) = 1 if machine \(i\) is assigned to task \(j\); 0 otherwise

##### Objective Function  
\[
\min \;\sum_{i\in M}\;\sum_{j\in T}\;c_{ij}\,x_{ij}
\]

##### Constraints  

###### 1. Machine‑to‑Task Assignment  
\[
\sum_{j\in T}x_{ij}=1,\qquad \forall i\in M
\]

###### 2. Task‑to‑Machine Assignment  
\[
\sum_{i\in M}x_{ij}=1,\qquad \forall j\in T
\]

###### 3. Binary Variables  
\[
x_{ij}\in\{0,1\},\qquad \forall i\in M,\; \forall j\in T
\]

##### Data Mapping  

| Parameter | Source Table | Source Columns |
|-----------|--------------|----------------|
| \(c_{ij}\) | `file_0_view_0` | `Machine`, `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L` |

*The value of \(c_{ij}\) is taken from the row where `Machine` equals \(i\) and the column corresponding to task \(j\).*