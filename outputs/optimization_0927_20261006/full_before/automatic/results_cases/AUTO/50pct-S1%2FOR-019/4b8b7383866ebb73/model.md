##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

Where:
- $I = \{\text{supply1}, \text{supply2}, \text{supply3}, \text{supply4}, \text{supply5}, \text{supply6}, \text{supply7}, \text{supply8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

Customer Demands (from customer_demand.csv, in source order):

\[
\begin{align*}
demand1 &: 9 \\
demand2 &: 66 \\
demand3 &: 56 \\
demand4 &: 17 \\
demand5 &: 43 \\
demand6 &: 62 \\
demand7 &: 10 \\
demand8 &: 37 \\
\end{align*}
\]

Supplier Capacities (from supply_capacity.csv, in source order):

\[
\begin{align*}
supply1 &: 60 \\
supply2 &: 22 \\
supply3 &: 16 \\
supply4 &: 14 \\
supply5 &: 19 \\
supply6 &: 70 \\
supply7 &: 60 \\
supply8 &: 39 \\
\end{align*}
\]

Transportation Costs (from transportation_costs.csv, in source order):

\[
\begin{array}{l|cccccccc}
 & demand1 & demand2 & demand3 & demand4 & demand5 & demand6 & demand7 & demand8 \\
\hline
supply1 & 0.03020736643461065 & 229.50723504640203 & 198.62356558205792 & 12.995050640153751 & 211.20732124396406 & 134.9442985029274 & 9.822206398831067 & 11.394077543225675 \\
supply2 & 232.34691308087835 & 3.6258726438627473 & 0.28605434149404785 & 45.73127693242935 & 2.8304796563034573 & 107.05891033185472 & 299.96317913389305 & 23.79935436307657 \\
supply3 & 11.061938334356302 & 0.2041995326579051 & 0.2789447278030927 & 45.721912724349636 & 59.54895565737313 & 5.097536739581239 & 300.00118415135785 & 23.711282707746893 \\
supply4 & 235.1794835706472 & 43.794668963036194 & 40.709846782945924 & 0.07774496620087613 & 4.237728183419554 & 131.70915517494691 & 296.55587567706743 & 29.810940017561297 \\
supply5 & 211.85808746383796 & 47.60180876530328 & 50.04007716193931 & 86.14548807358399 & 0.06197897916874956 & 5.3345515296262205 & 270.06290423798396 & 3.853933133973331 \\
supply6 & 6.45506633554524 & 88.16323623354015 & 5.047091671641611 & 151.46120287365497 & 5.290760161059401 & 0.04602205335871525 & 9.93670660180487 & 103.75460989446313 \\
supply7 & 174.27229047340035 & 250.58223528739327 & 253.90413041857263 & 16.235467318386764 & 12.643140514778086 & 175.0672824108511 & 2.983839625303656 & 317.0655193866389 \\
supply8 & 207.87006253790491 & 1.517168471518212 & 24.027239288137153 & 27.133999276450346 & 73.20672468851855 & 125.72910359893308 & 15.463103251642147 & 0.20164987511903337 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

where $c_{ij}$ is the transportation cost from supplier $i$ to customer $j$ as given above.

##### Constraints

1. **Demand Satisfaction** (each customer group must receive at least its demand):

   For each $j \in J$:
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]

2. **Supply Capacity** (each supplier cannot ship more than its capacity):

   For each $i \in I$:
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]

3. **Non-negativity**:

   For all $i \in I$, $j \in J$:
   \[
   x_{ij} \geq 0
   \]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
\end{align*}
\]

with all parameters and indices as specified above.