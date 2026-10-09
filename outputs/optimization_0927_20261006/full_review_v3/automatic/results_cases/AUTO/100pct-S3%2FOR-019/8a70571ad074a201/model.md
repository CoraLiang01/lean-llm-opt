##### Sets

Let  
$S = \{\text{supply1}, \text{supply2}, \text{supply3}, \text{supply4}, \text{supply5}, \text{supply6}, \text{supply7}, \text{supply8}\}$ (distribution centers)  
$D = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$ (customer groups)

##### Parameters

Demands (from customer_demand.csv, in source order):  
$d_{\text{demand1}} = 9$  
$d_{\text{demand2}} = 66$  
$d_{\text{demand3}} = 56$  
$d_{\text{demand4}} = 17$  
$d_{\text{demand5}} = 43$  
$d_{\text{demand6}} = 62$  
$d_{\text{demand7}} = 10$  
$d_{\text{demand8}} = 37$  

Supply capacities (from supply_capacity.csv, in source order):  
$s_{\text{supplier1}} = 60$  
$s_{\text{supplier2}} = 22$  
$s_{\text{supplier3}} = 16$  
$s_{\text{supplier4}} = 14$  
$s_{\text{supplier5}} = 19$  
$s_{\text{supplier6}} = 70$  
$s_{\text{supplier7}} = 60$  
$s_{\text{supplier8}} = 39$  

Transportation costs $c_{ij}$ (from transportation_costs.csv, in source order):

|           | demand1           | demand2           | demand3           | demand4           | demand5           | demand6           | demand7           | demand8           |
|-----------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| supply1   | 0.03020736643461065 | 229.50723504640203 | 198.62356558205792 | 12.995050640153751 | 211.20732124396406 | 134.9442985029274 | 9.822206398831067 | 11.394077543225675 |
| supply2   | 232.34691308087835 | 3.6258726438627473 | 0.28605434149404785 | 45.73127693242935 | 2.8304796563034573 | 107.05891033185472 | 299.96317913389305 | 23.79935436307657 |
| supply3   | 11.061938334356302 | 0.2041995326579051 | 0.2789447278030927 | 45.721912724349636 | 59.54895565737313 | 5.097536739581239 | 300.00118415135785 | 23.711282707746893 |
| supply4   | 235.1794835706472 | 43.794668963036194 | 40.709846782945924 | 0.07774496620087613 | 4.237728183419554 | 131.70915517494691 | 296.55587567706743 | 29.810940017561297 |
| supply5   | 211.85808746383796 | 47.60180876530328 | 50.04007716193931 | 86.14548807358399 | 0.06197897916874956 | 5.3345515296262205 | 270.06290423798396 | 3.853933133973331 |
| supply6   | 6.45506633554524 | 88.16323623354015 | 5.047091671641611 | 151.46120287365497 | 5.290760161059401 | 0.04602205335871525 | 9.93670660180487 | 103.75460989446313 |
| supply7   | 174.27229047340035 | 250.58223528739327 | 253.90413041857263 | 16.235467318386764 | 12.643140514778086 | 175.0672824108511 | 2.983839625303656 | 317.0655193866389 |
| supply8   | 207.87006253790491 | 1.517168471518212 | 24.027239288137153 | 27.133999276450346 | 73.20672468851855 | 125.72910359893308 | 15.463103251642147 | 0.20164987511903337 |

##### Decision Variables

For each $i \in S$, $j \in D$:

$x_{ij} \geq 0$ = quantity shipped from supplier $i$ to customer group $j$ (continuous)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in S} \sum_{j \in D} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction (for each $j \in D$):
$$
\sum_{i \in S} x_{ij} \geq d_j
$$

2. Supply capacity (for each $i \in S$):
$$
\sum_{j \in D} x_{ij} \leq s_i
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in S,\, j \in D
$$

##### Complete Numerical Formulation

Minimize
\[
\begin{align*}
&0.03020736643461065\,x_{\text{supply1},\text{demand1}} + 229.50723504640203\,x_{\text{supply1},\text{demand2}} + 198.62356558205792\,x_{\text{supply1},\text{demand3}} \\
&+ 12.995050640153751\,x_{\text{supply1},\text{demand4}} + 211.20732124396406\,x_{\text{supply1},\text{demand5}} + 134.9442985029274\,x_{\text{supply1},\text{demand6}} \\
&+ 9.822206398831067\,x_{\text{supply1},\text{demand7}} + 11.394077543225675\,x_{\text{supply1},\text{demand8}} \\
&+ 232.34691308087835\,x_{\text{supply2},\text{demand1}} + 3.6258726438627473\,x_{\text{supply2},\text{demand2}} + 0.28605434149404785\,x_{\text{supply2},\text{demand3}} \\
&+ 45.73127693242935\,x_{\text{supply2},\text{demand4}} + 2.8304796563034573\,x_{\text{supply2},\text{demand5}} + 107.05891033185472\,x_{\text{supply2},\text{demand6}} \\
&+ 299.96317913389305\,x_{\text{supply2},\text{demand7}} + 23.79935436307657\,x_{\text{supply2},\text{demand8}} \\
&+ 11.061938334356302\,x_{\text{supply3},\text{demand1}} + 0.2041995326579051\,x_{\text{supply3},\text{demand2}} + 0.2789447278030927\,x_{\text{supply3},\text{demand3}} \\
&+ 45.721912724349636\,x_{\text{supply3},\text{demand4}} + 59.54895565737313\,x_{\text{supply3},\text{demand5}} + 5.097536739581239\,x_{\text{supply3},\text{demand6}} \\
&+ 300.00118415135785\,x_{\text{supply3},\text{demand7}} + 23.711282707746893\,x_{\text{supply3},\text{demand8}} \\
&+ 235.1794835706472\,x_{\text{supply4},\text{demand1}} + 43.794668963036194\,x_{\text{supply4},\text{demand2}} + 40.709846782945924\,x_{\text{supply4},\text{demand3}} \\
&+ 0.07774496620087613\,x_{\text{supply4},\text{demand4}} + 4.237728183419554\,x_{\text{supply4},\text{demand5}} + 131.70915517494691\,x_{\text{supply4},\text{demand6}} \\
&+ 296.55587567706743\,x_{\text{supply4},\text{demand7}} + 29.810940017561297\,x_{\text{supply4},\text{demand8}} \\
&+ 211.85808746383796\,x_{\text{supply5},\text{demand1}} + 47.60180876530328\,x_{\text{supply5},\text{demand2}} + 50.04007716193931\,x_{\text{supply5},\text{demand3}} \\
&+ 86.14548807358399\,x_{\text{supply5},\text{demand4}} + 0.06197897916874956\,x_{\text{supply5},\text{demand5}} + 5.3345515296262205\,x_{\text{supply5},\text{demand6}} \\
&+ 270.06290423798396\,x_{\text{supply5},\text{demand7}} + 3.853933133973331\,x_{\text{supply5},\text{demand8}} \\
&+ 6.45506633554524\,x_{\text{supply6},\text{demand1}} + 88.16323623354015\,x_{\text{supply6},\text{demand2}} + 5.047091671641611\,x_{\text{supply6},\text{demand3}} \\
&+ 151.46120287365497\,x_{\text{supply6},\text{demand4}} + 5.290760161059401\,x_{\text{supply6},\text{demand5}} + 0.04602205335871525\,x_{\text{supply6},\text{demand6}} \\
&+ 9.93670660180487\,x_{\text{supply6},\text{demand7}} + 103.75460989446313\,x_{\text{supply6},\text{demand8}} \\
&+ 174.27229047340035\,x_{\text{supply7},\text{demand1}} + 250.58223528739327\,x_{\text{supply7},\text{demand2}} + 253.90413041857263\,x_{\text{supply7},\text{demand3}} \\
&+ 16.235467318386764\,x_{\text{supply7},\text{demand4}} + 12.643140514778086\,x_{\text{supply7},\text{demand5}} + 175.0672824108511\,x_{\text{supply7},\text{demand6}} \\
&+ 2.983839625303656\,x_{\text{supply7},\text{demand7}} + 317.0655193866389\,x_{\text{supply7},\text{demand8}} \\
&+ 207.87006253790491\,x_{\text{supply8},\text{demand1}} + 1.517168471518212\,x_{\text{supply8},\text{demand2}} + 24.027239288137153\,x_{\text{supply8},\text{demand3}} \\
&+ 27.133999276450346\,x_{\text{supply8},\text{demand4}} + 73.20672468851855\,x_{\text{supply8},\text{demand5}} + 125.72910359893308\,x_{\text{supply8},\text{demand6}} \\
&+ 15.463103251642147\,x_{\text{supply8},\text{demand7}} + 0.20164987511903337\,x_{\text{supply8},\text{demand8}}
\end{align*}
\]

Subject to:

For each customer group:
\[
\begin{align*}
x_{\text{supply1},\text{demand1}} + x_{\text{supply2},\text{demand1}} + x_{\text{supply3},\text{demand1}} + x_{\text{supply4},\text{demand1}} + x_{\text{supply5},\text{demand1}} + x_{\text{supply6},\text{demand1}} + x_{\text{supply7},\text{demand1}} + x_{\text{supply8},\text{demand1}} &\geq 9 \\
x_{\text{supply1},\text{demand2}} + x_{\text{supply2},\text{demand2}} + x_{\text{supply3},\text{demand2}} + x_{\text{supply4},\text{demand2}} + x_{\text{supply5},\text{demand2}} + x_{\text{supply6},\text{demand2}} + x_{\text{supply7},\text{demand2}} + x_{\text{supply8},\text{demand2}} &\geq 66 \\
x_{\text{supply1},\text{demand3}} + x_{\text{supply2},\text{demand3}} + x_{\text{supply3},\text{demand3}} + x_{\text{supply4},\text{demand3}} + x_{\text{supply5},\text{demand3}} + x_{\text{supply6},\text{demand3}} + x_{\text{supply7},\text{demand3}} + x_{\text{supply8},\text{demand3}} &\geq 56 \\
x_{\text{supply1},\text{demand4}} + x_{\text{supply2},\text{demand4}} + x_{\text{supply3},\text{demand4}} + x_{\text{supply4},\text{demand4}} + x_{\text{supply5},\text{demand4}} + x_{\text{supply6},\text{demand4}} + x_{\text{supply7},\text{demand4}} + x_{\text{supply8},\text{demand4}} &\geq 17 \\
x_{\text{supply1},\text{demand5}} + x_{\text{supply2},\text{demand5}} + x_{\text{supply3},\text{demand5}} + x_{\text{supply4},\text{demand5}} + x_{\text{supply5},\text{demand5}} + x_{\text{supply6},\text{demand5}} + x_{\text{supply7},\text{demand5}} + x_{\text{supply8},\text{demand5}} &\geq 43 \\
x_{\text{supply1},\text{demand6}} + x_{\text{supply2},\text{demand6}} + x_{\text{supply3},\text{demand6}} + x_{\text{supply4},\text{demand6}} + x_{\text{supply5},\text{demand6}} + x_{\text{supply6},\text{demand6}} + x_{\text{supply7},\text{demand6}} + x_{\text{supply8},\text{demand6}} &\geq 62 \\
x_{\text{supply1},\text{demand7}} + x_{\text{supply2},\text{demand7}} + x_{\text{supply3},\text{demand7}} + x_{\text{supply4},\text{demand7}} + x_{\text{supply5},\text{demand7}} + x_{\text{supply6},\text{demand7}} + x_{\text{supply7},\text{demand7}} + x_{\text{supply8},\text{demand7}} &\geq 10 \\
x_{\text{supply1},\text{demand8}} + x_{\text{supply2},\text{demand8}} + x_{\text{supply3},\text{demand8}} + x_{\text{supply4},\text{demand8}} + x_{\text{supply5},\text{demand8}} + x_{\text{supply6},\text{demand8}} + x_{\text{supply7},\text{demand8}} + x_{\text{supply8},\text{demand8}} &\geq 37 \\
\end{align*}
\]

For each supplier:
\[
\begin{align*}
x_{\text{supply1},\text{demand1}} + x_{\text{supply1},\text{demand2}} + x_{\text{supply1},\text{demand3}} + x_{\text{supply1},\text{demand4}} + x_{\text{supply1},\text{demand5}} + x_{\text{supply1},\text{demand6}} + x_{\text{supply1},\text{demand7}} + x_{\text{supply1},\text{demand8}} &\leq 60 \\
x_{\text{supply2},\text{demand1}} + x_{\text{supply2},\text{demand2}} + x_{\text{supply2},\text{demand3}} + x_{\text{supply2},\text{demand4}} + x_{\text{supply2},\text{demand5}} + x_{\text{supply2},\text{demand6}} + x_{\text{supply2},\text{demand7}} + x_{\text{supply2},\text{demand8}} &\leq 22 \\
x_{\text{supply3},\text{demand1}} + x_{\text{supply3},\text{demand2}} + x_{\text{supply3},\text{demand3}} + x_{\text{supply3},\text{demand4}} + x_{\text{supply3},\text{demand5}} + x_{\text{supply3},\text{demand6}} + x_{\text{supply3},\text{demand7}} + x_{\text{supply3},\text{demand8}} &\leq 16 \\
x_{\text{supply4},\text{demand1}} + x_{\text{supply4},\text{demand2}} + x_{\text{supply4},\text{demand3}} + x_{\text{supply4},\text{demand4}} + x_{\text{supply4},\text{demand5}} + x_{\text{supply4},\text{demand6}} + x_{\text{supply4},\text{demand7}} + x_{\text{supply4},\text{demand8}} &\leq 14 \\
x_{\text{supply5},\text{demand1}} + x_{\text{supply5},\text{demand2}} + x_{\text{supply5},\text{demand3}} + x_{\text{supply5},\text{demand4}} + x_{\text{supply5},\text{demand5}} + x_{\text{supply5},\text{demand6}} + x_{\text{supply5},\text{demand7}} + x_{\text{supply5},\text{demand8}} &\leq 19 \\
x_{\text{supply6},\text{demand1}} + x_{\text{supply6},\text{demand2}} + x_{\text{supply6},\text{demand3}} + x_{\text{supply6},\text{demand4}} + x_{\text{supply6},\text{demand5}} + x_{\text{supply6},\text{demand6}} + x_{\text{supply6},\text{demand7}} + x_{\text{supply6},\text{demand8}} &\leq 70 \\
x_{\text{supply7},\text{demand1}} + x_{\text{supply7},\text{demand2}} + x_{\text{supply7},\text{demand3}} + x_{\text{supply7},\text{demand4}} + x_{\text{supply7},\text{demand5}} + x_{\text{supply7},\text{demand6}} + x_{\text{supply7},\text{demand7}} + x_{\text{supply7},\text{demand8}} &\leq 60 \\
x_{\text{supply8},\text{demand1}} + x_{\text{supply8},\text{demand2}} + x_{\text{supply8},\text{demand3}} + x_{\text{supply8},\text{demand4}} + x_{\text{supply8},\text{demand5}} + x_{\text{supply8},\text{demand6}} + x_{\text{supply8},\text{demand7}} + x_{\text{supply8},\text{demand8}} &\leq 39 \\
\end{align*}
\]

And for all $i \in S$, $j \in D$:
\[
x_{ij} \geq 0
\]