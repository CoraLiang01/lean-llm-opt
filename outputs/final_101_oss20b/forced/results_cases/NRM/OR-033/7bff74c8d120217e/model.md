##### Index Set  
Let  
\[
I \;=\; \{\, i \mid \text{Product } i \text{ is classified as } \text{Baby} \,\}.
\]

##### Parameters  
For each \(i \in I\) define  
\[
\begin{aligned}
w_i &\;=\; \text{Revenue of product } i,\\
d_i &\;=\; \text{Demand of product } i,\\
c_i &\;=\; \text{Initial Inventory of product } i.
\end{aligned}
\]

##### Variables  
\[
x_i \;\in\; \mathbb{Z}, \quad x_i \;\ge\; 0, \qquad \forall i \in I.
\]

##### Objective Function  
\[
\quad \max \;\sum_{i \in I} w_i \, x_i
\]

##### Constraints  

###### 1. Demand Constraints  
\[
x_i \;\le\; d_i, \qquad \forall i \in I
\]

###### 2. Inventory Constraints  
\[
x_i \;\le\; c_i, \qquad \forall i \in I
\]

##### Data Mapping  

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(w_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(c_i\)   | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | `file_0_view_0` | `Product Name` (filter: `Product Name` contains “Baby”) |

*Only the mapping of parameters to source columns is shown; record values and counts are omitted.*