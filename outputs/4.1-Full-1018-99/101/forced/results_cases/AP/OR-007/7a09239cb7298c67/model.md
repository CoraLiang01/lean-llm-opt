##### Decision Variables:

Let $x_{ij}$ denote the number of units shipped from warehouse $i$ to store $j$, where $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ and $j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$.

##### Objective Function:

$\min \sum_{i \in S} \sum_{j \in D} c_{ij} x_{ij}$

where $c_{ij}$ is the unit transportation cost from warehouse $i$ to store $j$.

##### Constraints:

###### 1. Demand Satisfaction (for each store):

$\sum_{i \in S} x_{ij} = d_j \quad \forall j \in D$

where $d_j$ is the demand at store $j$.

###### 2. Supply Capacity (for each warehouse):

$\sum_{j \in D} x_{ij} \leq s_i \quad \forall i \in S$

where $s_i$ is the supply capacity of warehouse $i$.

###### 3. Non-negativity:

$x_{ij} \geq 0 \quad \forall i \in S, \forall j \in D$

---

##### Retrieved Information

**Warehouses (Supply Nodes):**
- S1: supply_capacity = 428
- S2: supply_capacity = 217
- S3: supply_capacity = 214
- S4: supply_capacity = 380
- S5: supply_capacity = 254

**Stores (Demand Nodes):**
- D1: demand = 428
- D2: demand = 217
- D3: demand = 214
- D4: demand = 380
- D5: demand = 254

**Transportation Costs ($c_{ij}$):**

|        | D1           | D2           | D3           | D4           | D5           |
|--------|--------------|--------------|--------------|--------------|--------------|
| S1     | 269.3910588  | 1.45373354   | 99.60345346  | 26.64078166  | 9.53768896   |
| S2     | 9.29184688   | 10.87477844  | 144.5260929  | 11.42013308  | 153.17568199 |
| S3     | 9.67458430   | 2.61916510   | 100.82422492 | 3.21219109   | 133.84933961 |
| S4     | 270.5749848  | 32.50253586  | 4.68420981   | 1.56822697   | 9.58927599   |
| S5     | 226.03319107 | 8.66916198   | 65.47681317  | 9.06876526   | 202.65015316 |

**Sets:**
- $S = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $D = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

**Parameters:**
- $d_{\text{D1}} = 428$, $d_{\text{D2}} = 217$, $d_{\text{D3}} = 214$, $d_{\text{D4}} = 380$, $d_{\text{D5}} = 254$
- $s_{\text{S1}} = 428$, $s_{\text{S2}} = 217$, $s_{\text{S3}} = 214$, $s_{\text{S4}} = 380$, $s_{\text{S5}} = 254$
- $c_{ij}$ as in the table above

##### Complete Mathematical Model

$\min \Bigg[ $
$269.3910588\,x_{\text{S1},\text{D1}} + 1.45373354\,x_{\text{S1},\text{D2}} + 99.60345346\,x_{\text{S1},\text{D3}} + 26.64078166\,x_{\text{S1},\text{D4}} + 9.53768896\,x_{\text{S1},\text{D5}} +$
$9.29184688\,x_{\text{S2},\text{D1}} + 10.87477844\,x_{\text{S2},\text{D2}} + 144.5260929\,x_{\text{S2},\text{D3}} + 11.42013308\,x_{\text{S2},\text{D4}} + 153.17568199\,x_{\text{S2},\text{D5}} +$
$9.67458430\,x_{\text{S3},\text{D1}} + 2.61916510\,x_{\text{S3},\text{D2}} + 100.82422492\,x_{\text{S3},\text{D3}} + 3.21219109\,x_{\text{S3},\text{D4}} + 133.84933961\,x_{\text{S3},\text{D5}} +$
$270.5749848\,x_{\text{S4},\text{D1}} + 32.50253586\,x_{\text{S4},\text{D2}} + 4.68420981\,x_{\text{S4},\text{D3}} + 1.56822697\,x_{\text{S4},\text{D4}} + 9.58927599\,x_{\text{S4},\text{D5}} +$
$226.03319107\,x_{\text{S5},\text{D1}} + 8.66916198\,x_{\text{S5},\text{D2}} + 65.47681317\,x_{\text{S5},\text{D3}} + 9.06876526\,x_{\text{S5},\text{D4}} + 202.65015316\,x_{\text{S5},\text{D5}}$
$\Bigg]$

Subject to:

$\begin{align*}
x_{\text{S1},\text{D1}} + x_{\text{S2},\text{D1}} + x_{\text{S3},\text{D1}} + x_{\text{S4},\text{D1}} + x_{\text{S5},\text{D1}} &= 428 \\
x_{\text{S1},\text{D2}} + x_{\text{S2},\text{D2}} + x_{\text{S3},\text{D2}} + x_{\text{S4},\text{D2}} + x_{\text{S5},\text{D2}} &= 217 \\
x_{\text{S1},\text{D3}} + x_{\text{S2},\text{D3}} + x_{\text{S3},\text{D3}} + x_{\text{S4},\text{D3}} + x_{\text{S5},\text{D3}} &= 214 \\
x_{\text{S1},\text{D4}} + x_{\text{S2},\text{D4}} + x_{\text{S3},\text{D4}} + x_{\text{S4},\text{D4}} + x_{\text{S5},\text{D4}} &= 380 \\
x_{\text{S1},\text{D5}} + x_{\text{S2},\text{D5}} + x_{\text{S3},\text{D5}} + x_{\text{S4},\text{D5}} + x_{\text{S5},\text{D5}} &= 254 \\
x_{\text{S1},\text{D1}} + x_{\text{S1},\text{D2}} + x_{\text{S1},\text{D3}} + x_{\text{S1},\text{D4}} + x_{\text{S1},\text{D5}} &\leq 428 \\
x_{\text{S2},\text{D1}} + x_{\text{S2},\text{D2}} + x_{\text{S2},\text{D3}} + x_{\text{S2},\text{D4}} + x_{\text{S2},\text{D5}} &\leq 217 \\
x_{\text{S3},\text{D1}} + x_{\text{S3},\text{D2}} + x_{\text{S3},\text{D3}} + x_{\text{S3},\text{D4}} + x_{\text{S3},\text{D5}} &\leq 214 \\
x_{\text{S4},\text{D1}} + x_{\text{S4},\text{D2}} + x_{\text{S4},\text{D3}} + x_{\text{S4},\text{D4}} + x_{\text{S4},\text{D5}} &\leq 380 \\
x_{\text{S5},\text{D1}} + x_{\text{S5},\text{D2}} + x_{\text{S5},\text{D3}} + x_{\text{S5},\text{D4}} + x_{\text{S5},\text{D5}} &\leq 254 \\
x_{ij} &\geq 0 \quad \forall i \in S, j \in D
\end{align*}$

Where all parameters and variables are as defined above.