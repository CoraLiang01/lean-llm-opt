##### Objective Function: 

$\quad \quad \max \quad \sum_{b \in B} P_b \, y_b \;-\; \sum_{g \in G} C_g \, \sum_{b \in B} x_{b,g}$

##### Constraints

###### 1. Blending Proportion Constraints: 

$\quad \quad L_{b,g}\, y_b \;\le\; x_{b,g} \;\le\; U_{b,g}\, y_b , \qquad \forall\, b \in B,\; g \in G$

###### 2. Raw Material Supply Constraints: 

$\quad \quad \sum_{b \in B} x_{b,g} \;\le\; S_g , \qquad \forall\, g \in G$

###### 3. Production Definition: 

$\quad \quad y_b \;=\; \sum_{g \in G} x_{b,g} , \qquad \forall\, b \in B$

###### 4. Minimum Production for Red Brand: 

$\quad \quad y_{\text{Red}} \;\ge\; 2000$

###### 5. Non‑negativity: 

$\quad \quad x_{b,g} \;\ge\; 0 , \quad y_b \;\ge\; 0 , \qquad \forall\, b \in B,\; g \in G$

##### Data Mapping

| Table ID | Column Name | Parameter |
|----------|-------------|-----------|
| `file_0_view_0` | `Grade` | Index \(g\) |
| `file_0_view_0` | `Daily Supply (kg)` | \(S_g\) |
| `file_0_view_0` | `Cost (CNY/kg)` | \(C_g\) |
| `file_1_view_0` | `Brand` | Index \(b\) |
| `file_1_view_0` | `Selling Price (CNY/kg)` | \(P_b\) |
| `file_1_view_0` | `Blending Requirements` | Defines \(L_{b,g}\) and \(U_{b,g}\) for each \(b,g\) |

*All indices and parameters are defined over the sets \(G\) and \(B\) extracted from the respective tables.*