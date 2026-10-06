##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, where $i \in I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$ and $j \in J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$.

##### Parameters

Demands:
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

Supply capacities:
\[
\begin{align*}
supplier1 &: 60 \\
supplier2 &: 22 \\
supplier3 &: 16 \\
supplier4 &: 14 \\
supplier5 &: 19 \\
supplier6 &: 70 \\
supplier7 &: 60 \\
supplier8 &: 39 \\
\end{align*}
\]

Transportation costs $c_{ij}$ (rows: suppliers, columns: demands):

\[
\begin{array}{l|cccccccc}
 & demand1 & demand2 & demand3 & demand4 & demand5 & demand6 & demand7 & demand8 \\
\hline
supply1 & 0.0302073664346106 & 229.50723504640203 & 198.62356558205792 & 12.995050640153751 & 211.2073212439641 & 134.9442985029274 & 9.822206398831067 & 11.394077543225675 \\
supply2 & 232.34691308087835 & 3.6258726438627473 & 0.2860543414940478 & 45.73127693242935 & 2.8304796563034573 & 107.05891033185472 & 299.96317913389305 & 23.79935436307657 \\
supply3 & 11.061938334356302 & 0.2041995326579051 & 0.2789447278030927 & 45.72191272434964 & 59.54895565737313 & 5.097536739581239 & 300.00118415135785 & 23.711282707746893 \\
supply4 & 235.1794835706472 & 43.794668963036194 & 40.70984678294592 & 0.0777449662008761 & 4.237728183419554 & 131.70915517494691 & 296.55587567706743 & 29.810940017561297 \\
supply5 & 211.858087463838 & 47.60180876530328 & 50.04007716193931 & 86.14548807358399 & 0.0619789791687495 & 5.3345515296262205 & 270.062904237984 & 3.853933133973331 \\
supply6 & 6.45506633554524 & 88.16323623354015 & 5.047091671641611 & 151.46120287365497 & 5.290760161059401 & 0.0460220533587152 & 9.93670660180487 & 103.75460989446312 \\
supply7 & 174.27229047340035 & 250.58223528739327 & 253.90413041857263 & 16.235467318386764 & 12.643140514778086 & 175.0672824108511 & 2.983839625303656 & 317.0655193866389 \\
supply8 & 207.8700625379049 & 1.517168471518212 & 24.027239288137157 & 27.133999276450343 & 73.20672468851855 & 125.72910359893308 & 15.463103251642147 & 0.2016498751190333 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. Demand satisfaction (for each customer group $j$):
\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]

2. Supply capacity (for each supplier $i$):
\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]

3. Non-negativity:
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

##### Sets

\[
I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}
\]
\[
J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}
\]