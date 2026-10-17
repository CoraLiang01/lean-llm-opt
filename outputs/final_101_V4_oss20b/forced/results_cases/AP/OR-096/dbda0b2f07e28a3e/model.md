##### Sets  

\[
S=\{I,\,II\}, \qquad N=\{N01,N02,\dots ,N31\}
\]

##### Parameters  

\[
\begin{aligned}
\text{Cap}_s &\;=\;\text{Capacity of school }s &&\text{from } \texttt{school_capacity.csv} \\
\text{PopWhite}_n &\;=\;\text{Population of white pupils in neighborhood }n &&\text{from } \texttt{neighborhoods_population.csv} \\
\text{PopNonWhite}_n &\;=\;\text{Population of non‑white pupils in neighborhood }n &&\text{from } \texttt{neighborhoods_population.csv} \\
\text{Dist}_{s,n} &\;=\;\text{Distance (miles) from school }s\text{ to neighborhood }n &&\text{from } \texttt{distance.csv}
\end{aligned}
\]

##### Decision Variables  

\[
\begin{aligned}
x_{s,n}^{W} &\in \mathbb{Z}_{+} &&\text{number of white pupils from }n\text{ assigned to }s \\
x_{s,n}^{NW} &\in \mathbb{Z}_{+} &&\text{number of non‑white pupils from }n\text{ assigned to }s
\end{aligned}
\]

##### Objective  

\[
\min \;\sum_{s\in S}\sum_{n\in N}\text{Dist}_{s,n}\,\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)
\]

##### Constraints  

1. **School capacity**  
\[
\sum_{n\in N}\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)\;\le\;\text{Cap}_s
\qquad \forall s\in S
\]

2. **Neighborhood population**  
\[
\sum_{s\in S}x_{s,n}^{W}\;=\;\text{PopWhite}_n
\qquad \forall n\in N
\]
\[
\sum_{s\in S}x_{s,n}^{NW}\;=\;\text{PopNonWhite}_n
\qquad \forall n\in N
\]

3. **Racial balance**  
Let  
\[
W_s=\sum_{n\in N}x_{s,n}^{W}, \qquad
T_s=\sum_{n\in N}\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)
\]
Then  
\[
0.5\,T_s \;\le\; W_s \;\le\; 0.7\,T_s
\qquad \forall s\in S
\]

##### Data Mapping  

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| `Cap_s` | `file_0_view_0` | `Capacity` | `School` → `Cap_s` |
| `PopWhite_n` | `file_1_view_0` | `Population_White` | `Neighborhood` → `PopWhite_n` |
| `PopNonWhite_n` | `file_1_view_0` | `Population_NonWhite` | `Neighborhood` → `PopNonWhite_n` |
| `Dist_{s,n}` | `file_2_view_0` | `N01`–`N31` | `School` → row, `Neighborhood` → column |

All indices, variables, and constraints are defined solely from the data and the problem statement.