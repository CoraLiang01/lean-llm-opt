##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- Demands:
  - $d_{\text{demand1}} = 9$
  - $d_{\text{demand2}} = 66$
  - $d_{\text{demand3}} = 56$
  - $d_{\text{demand4}} = 17$
  - $d_{\text{demand5}} = 43$
  - $d_{\text{demand6}} = 62$
  - $d_{\text{demand7}} = 10$
  - $d_{\text{demand8}} = 37$

- Supply capacities:
  - $s_{\text{supplier1}} = 60$
  - $s_{\text{supplier2}} = 22$
  - $s_{\text{supplier3}} = 16$
  - $s_{\text{supplier4}} = 14$
  - $s_{\text{supplier5}} = 19$
  - $s_{\text{supplier6}} = 70$
  - $s_{\text{supplier7}} = 60$
  - $s_{\text{supplier8}} = 39$

- Transportation costs $c_{ij}$:

|            | demand1         | demand2         | demand3         | demand4         | demand5         | demand6         | demand7         | demand8         |
|------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| supplier1  | 0.03020736643461065 | 229.50723504640203 | 198.62356558205792 | 12.995050640153751 | 211.20732124396406 | 134.9442985029274 | 9.822206398831067 | 11.394077543225675 |
| supplier2  | 232.34691308087835 | 3.6258726438627473 | 0.28605434149404785 | 45.73127693242935 | 2.8304796563034573 | 107.05891033185472 | 299.96317913389305 | 23.79935436307657 |
| supplier3  | 11.061938334356302 | 0.2041995326579051 | 0.2789447278030927 | 45.721912724349636 | 59.54895565737313 | 5.097536739581239 | 300.00118415135785 | 23.711282707746893 |
| supplier4  | 235.1794835706472 | 43.794668963036194 | 40.709846782945924 | 0.07774496620087613 | 4.237728183419554 | 131.70915517494691 | 296.55587567706743 | 29.810940017561297 |
| supplier5  | 211.85808746383796 | 47.60180876530328 | 50.04007716193931 | 86.14548807358399 | 0.06197897916874956 | 5.3345515296262205 | 270.06290423798396 | 3.853933133973331 |
| supplier6  | 6.45506633554524 | 88.16323623354015 | 5.047091671641611 | 151.46120287365497 | 5.290760161059401 | 0.04602205335871525 | 9.93670660180487 | 103.75460989446313 |
| supplier7  | 174.27229047340035 | 250.58223528739327 | 253.90413041857263 | 16.235467318386764 | 12.643140514778086 | 175.0672824108511 | 2.983839625303656 | 317.0655193866389 |
| supplier8  | 207.87006253790491 | 1.517168471518212 | 24.027239288137153 | 27.133999276450346 | 73.20672468851855 | 125.72910359893308 | 15.463103251642147 | 0.20164987511903337 |

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$

2. **Supply capacity** (each supplier does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$

3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Full Numerical Model

Let $x_{ij}$ be the amount shipped from supplier $i$ to customer $j$.

**Minimize:**
\[
\begin{align*}
\min\ & \sum_{j=1}^8 \Big[ 
  0.03020736643461065\, x_{\text{supplier1},j=1} + 229.50723504640203\, x_{\text{supplier1},j=2} + 198.62356558205792\, x_{\text{supplier1},j=3} + 12.995050640153751\, x_{\text{supplier1},j=4} \\
&\quad + 211.20732124396406\, x_{\text{supplier1},j=5} + 134.9442985029274\, x_{\text{supplier1},j=6} + 9.822206398831067\, x_{\text{supplier1},j=7} + 11.394077543225675\, x_{\text{supplier1},j=8} \\
&\quad + 232.34691308087835\, x_{\text{supplier2},j=1} + 3.6258726438627473\, x_{\text{supplier2},j=2} + 0.28605434149404785\, x_{\text{supplier2},j=3} + 45.73127693242935\, x_{\text{supplier2},j=4} \\
&\quad + 2.8304796563034573\, x_{\text{supplier2},j=5} + 107.05891033185472\, x_{\text{supplier2},j=6} + 299.96317913389305\, x_{\text{supplier2},j=7} + 23.79935436307657\, x_{\text{supplier2},j=8} \\
&\quad + 11.061938334356302\, x_{\text{supplier3},j=1} + 0.2041995326579051\, x_{\text{supplier3},j=2} + 0.2789447278030927\, x_{\text{supplier3},j=3} + 45.721912724349636\, x_{\text{supplier3},j=4} \\
&\quad + 59.54895565737313\, x_{\text{supplier3},j=5} + 5.097536739581239\, x_{\text{supplier3},j=6} + 300.00118415135785\, x_{\text{supplier3},j=7} + 23.711282707746893\, x_{\text{supplier3},j=8} \\
&\quad + 235.1794835706472\, x_{\text{supplier4},j=1} + 43.794668963036194\, x_{\text{supplier4},j=2} + 40.709846782945924\, x_{\text{supplier4},j=3} + 0.07774496620087613\, x_{\text{supplier4},j=4} \\
&\quad + 4.237728183419554\, x_{\text{supplier4},j=5} + 131.70915517494691\, x_{\text{supplier4},j=6} + 296.55587567706743\, x_{\text{supplier4},j=7} + 29.810940017561297\, x_{\text{supplier4},j=8} \\
&\quad + 211.85808746383796\, x_{\text{supplier5},j=1} + 47.60180876530328\, x_{\text{supplier5},j=2} + 50.04007716193931\, x_{\text{supplier5},j=3} + 86.14548807358399\, x_{\text{supplier5},j=4} \\
&\quad + 0.06197897916874956\, x_{\text{supplier5},j=5} + 5.3345515296262205\, x_{\text{supplier5},j=6} + 270.06290423798396\, x_{\text{supplier5},j=7} + 3.853933133973331\, x_{\text{supplier5},j=8} \\
&\quad + 6.45506633554524\, x_{\text{supplier6},j=1} + 88.16323623354015\, x_{\text{supplier6},j=2} + 5.047091671641611\, x_{\text{supplier6},j=3} + 151.46120287365497\, x_{\text{supplier6},j=4} \\
&\quad + 5.290760161059401\, x_{\text{supplier6},j=5} + 0.04602205335871525\, x_{\text{supplier6},j=6} + 9.93670660180487\, x_{\text{supplier6},j=7} + 103.75460989446313\, x_{\text{supplier6},j=8} \\
&\quad + 174.27229047340035\, x_{\text{supplier7},j=1} + 250.58223528739327\, x_{\text{supplier7},j=2} + 253.90413041857263\, x_{\text{supplier7},j=3} + 16.235467318386764\, x_{\text{supplier7},j=4} \\
&\quad + 12.643140514778086\, x_{\text{supplier7},j=5} + 175.0672824108511\, x_{\text{supplier7},j=6} + 2.983839625303656\, x_{\text{supplier7},j=7} + 317.0655193866389\, x_{\text{supplier7},j=8} \\
&\quad + 207.87006253790491\, x_{\text{supplier8},j=1} + 1.517168471518212\, x_{\text{supplier8},j=2} + 24.027239288137153\, x_{\text{supplier8},j=3} + 27.133999276450346\, x_{\text{supplier8},j=4} \\
&\quad + 73.20672468851855\, x_{\text{supplier8},j=5} + 125.72910359893308\, x_{\text{supplier8},j=6} + 15.463103251642147\, x_{\text{supplier8},j=7} + 0.20164987511903337\, x_{\text{supplier8},j=8}
\Big]
\end{align*}
\]

**Subject to:**

For each customer $j$:
\[
\begin{align*}
x_{\text{supplier1},j} + x_{\text{supplier2},j} + x_{\text{supplier3},j} + x_{\text{supplier4},j} + x_{\text{supplier5},j} + x_{\text{supplier6},j} + x_{\text{supplier7},j} + x_{\text{supplier8},j} \geq d_j
\end{align*}
\]
where $d_j$ is as listed above for each $j$.

For each supplier $i$:
\[
\begin{align*}
x_{i,\text{demand1}} + x_{i,\text{demand2}} + x_{i,\text{demand3}} + x_{i,\text{demand4}} + x_{i,\text{demand5}} + x_{i,\text{demand6}} + x_{i,\text{demand7}} + x_{i,\text{demand8}} \leq s_i
\end{align*}
\]
where $s_i$ is as listed above for each $i$.

And for all $i, j$:
\[
x_{ij} \geq 0
\]

All identifiers and coefficients are preserved exactly as retrieved.