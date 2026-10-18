##### Sets
- $I = \{\, i \mid i \text{ is a ShelfID in } \texttt{file\_0\_view\_0}\,\}$
- $J = \{\, j \mid j \text{ is a ProductName in } \texttt{file\_1\_view\_0}\,\}$  

##### Parameters
- $Capacity_i = \text{Capacity of shelf } i \text{ from column } \texttt{Capacity} \text{ in } \texttt{file\_0\_view\_0}$
- $Value_j = \text{Value of product } j \text{ from column } \texttt{Value} \text{ in } \texttt{file\_1\_view\_0}$
- $Weight_j = \text{Weight of product } j \text{ from column } \texttt{Weight} \text{ in } \texttt{file\_1\_view\_0}$  

##### Variables
- $x_{ij} \in \mathbb{Z},\; x_{ij} \ge 0 \quad \forall i \in I,\; \forall j \in J$  

##### Objective Function
$ \displaystyle \max \sum_{i \in I} \sum_{j \in J} Value_j \cdot x_{ij}$  

##### Constraints
###### Capacity constraints  
$ \displaystyle \sum_{j \in J} Weight_j \cdot x_{ij} \;\le\; Capacity_i \quad \forall i \in I$  

###### Non‑negativity and integrality  
$ x_{ij} \in \mathbb{Z},\; x_{ij} \ge 0 \quad \forall i \in I,\; \forall j \in J$  

##### Data Mapping
- **Table** `file_0_view_0`: columns `ShelfID`, `Capacity`  
- **Table** `file_1_view_0`: columns `ProductName`, `Value`, `Weight`