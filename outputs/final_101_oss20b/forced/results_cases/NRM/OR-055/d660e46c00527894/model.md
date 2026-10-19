##### Index Sets  

\[
I \;=\; \{\, i \mid \text{DisplayID } i \text{ appears in } \texttt{capacity.csv}\,\}
\]

\[
J \;=\; \{\, j \mid \text{ProductName } j \text{ appears in } \texttt{products.csv}\,\}
\]

##### Parameters  

\[
\text{capacity}_i \;=\; \text{Capacity of display area } i \quad (i \in I)
\]

\[
\text{weight}_j \;=\; \text{Weight of boat type } j \quad (j \in J)
\]

\[
\text{value}_j \;=\; \text{Value of boat type } j \quad (j \in J)
\]

##### Decision Variables  

\[
x_{ij} \;\in\; \mathbb{Z}_{\ge 0} \quad \forall\, i \in I,\; j \in J
\]

##### Objective Function  

\[
\max \;\sum_{i \in I}\;\sum_{j \in J}\; \text{value}_j \; x_{ij}
\]

##### Constraints  

###### Capacity Constraints  

\[
\sum_{j \in J}\; \text{weight}_j \; x_{ij} \;\le\; \text{capacity}_i
\quad \forall\, i \in I
\]

##### Data Mapping  

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| \(\text{capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{weight}_j\)   | `file_1_view_0` | `Weight`   |
| \(\text{value}_j\)    | `file_1_view_0` | `Value`    |

*No record values or counts are included; the model is expressed symbolically.*