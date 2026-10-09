Let:
- \( x_{ij} \) = quantity shipped from supplier \( i \) to customer \( j \), for all suppliers \( i \) and customers \( j \).

Sets:
- Suppliers: supplier1, supplier2, supplier3, supplier4, supplier5, supplier6, supplier7, supplier8
- Customers: demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8

Parameters (from CSVs, in source order):

Customer Demands:
\[
\begin{align*}
\text{demand1:} &\quad 9 \\
\text{demand2:} &\quad 66 \\
\text{demand3:} &\quad 56 \\
\text{demand4:} &\quad 17 \\
\text{demand5:} &\quad 43 \\
\text{demand6:} &\quad 62 \\
\text{demand7:} &\quad 10 \\
\text{demand8:} &\quad 37 \\
\end{align*}
\]

Supplier Capacities:
\[
\begin{align*}
\text{supplier1:} &\quad 60 \\
\text{supplier2:} &\quad 22 \\
\text{supplier3:} &\quad 16 \\
\text{supplier4:} &\quad 14 \\
\text{supplier5:} &\quad 19 \\
\text{supplier6:} &\quad 70 \\
\text{supplier7:} &\quad 60 \\
\text{supplier8:} &\quad 39 \\
\end{align*}
\]

Transportation Costs (\( c_{ij} \)), where \( i \) is supplier and \( j \) is customer:

\[
\begin{array}{l|cccccccc}
 & \text{demand1} & \text{demand2} & \text{demand3} & \text{demand4} & \text{demand5} & \text{demand6} & \text{demand7} & \text{demand8} \\
\hline
\text{supply1} & 0.03020736643461065 & 229.50723504640203 & 198.62356558205792 & 12.995050640153751 & 211.20732124396406 & 134.9442985029274 & 9.822206398831067 & 11.394077543225675 \\
\text{supply2} & 232.34691308087835 & 3.6258726438627473 & 0.28605434149404785 & 45.73127693242935 & 2.8304796563034573 & 107.05891033185472 & 299.96317913389305 & 23.79935436307657 \\
\text{supply3} & 11.061938334356302 & 0.2041995326579051 & 0.2789447278030927 & 45.721912724349636 & 59.54895565737313 & 5.097536739581239 & 300.00118415135785 & 23.711282707746893 \\
\text{supply4} & 235.1794835706472 & 43.794668963036194 & 40.709846782945924 & 0.07774496620087613 & 4.237728183419554 & 131.70915517494691 & 296.55587567706743 & 29.810940017561297 \\
\text{supply5} & 211.85808746383796 & 47.60180876530328 & 50.04007716193931 & 86.14548807358399 & 0.06197897916874956 & 5.3345515296262205 & 270.06290423798396 & 3.853933133973331 \\
\text{supply6} & 6.45506633554524 & 88.16323623354015 & 5.047091671641611 & 151.46120287365497 & 5.290760161059401 & 0.04602205335871525 & 9.93670660180487 & 103.75460989446313 \\
\text{supply7} & 174.27229047340035 & 250.58223528739327 & 253.90413041857263 & 16.235467318386764 & 12.643140514778086 & 175.0672824108511 & 2.983839625303656 & 317.0655193866389 \\
\text{supply8} & 207.87006253790491 & 1.517168471518212 & 24.027239288137153 & 27.133999276450346 & 73.20672468851855 & 125.72910359893308 & 15.463103251642147 & 0.20164987511903337 \\
\end{array}
\]

Decision Variables:
\[
x_{ij} \geq 0 \quad \forall i \in \{\text{supplier1},\ldots,\text{supplier8}\},\ j \in \{\text{demand1},\ldots,\text{demand8}\}
\]

Objective:
\[
\min \sum_{i=1}^{8} \sum_{j=1}^{8} c_{ij} x_{ij}
\]
where \( c_{ij} \) are as given above.

Constraints:

1. Demand satisfaction (for each customer \( j \)):
\[
\sum_{i=1}^{8} x_{ij} = \text{demand}_j \quad \forall j \in \{\text{demand1},\ldots,\text{demand8}\}
\]
That is:
\[
\begin{align*}
\sum_{i=1}^{8} x_{i,1} &= 9 \\
\sum_{i=1}^{8} x_{i,2} &= 66 \\
\sum_{i=1}^{8} x_{i,3} &= 56 \\
\sum_{i=1}^{8} x_{i,4} &= 17 \\
\sum_{i=1}^{8} x_{i,5} &= 43 \\
\sum_{i=1}^{8} x_{i,6} &= 62 \\
\sum_{i=1}^{8} x_{i,7} &= 10 \\
\sum_{i=1}^{8} x_{i,8} &= 37 \\
\end{align*}
\]

2. Supply capacity (for each supplier \( i \)):
\[
\sum_{j=1}^{8} x_{ij} \leq \text{supply\_capacity}_i \quad \forall i \in \{\text{supplier1},\ldots,\text{supplier8}\}
\]
That is:
\[
\begin{align*}
\sum_{j=1}^{8} x_{1,j} &\leq 60 \\
\sum_{j=1}^{8} x_{2,j} &\leq 22 \\
\sum_{j=1}^{8} x_{3,j} &\leq 16 \\
\sum_{j=1}^{8} x_{4,j} &\leq 14 \\
\sum_{j=1}^{8} x_{5,j} &\leq 19 \\
\sum_{j=1}^{8} x_{6,j} &\leq 70 \\
\sum_{j=1}^{8} x_{7,j} &\leq 60 \\
\sum_{j=1}^{8} x_{8,j} &\leq 39 \\
\end{align*}
\]

3. Nonnegativity:
\[
x_{ij} \geq 0 \quad \forall i, j
\]

This is a complete numerical linear programming formulation for the described transportation problem, preserving all identifiers, coefficients, and source order.