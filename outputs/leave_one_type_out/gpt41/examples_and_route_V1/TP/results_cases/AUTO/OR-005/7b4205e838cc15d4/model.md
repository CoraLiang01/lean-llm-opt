Let $x_{ij}$ be the number of units shipped from supplier $i$ to customer $j$, where $i$ indexes the suppliers and $j$ indexes the customer groups. All $x_{ij} \geq 0$ and integer.

Minimize total transportation cost:
$$
\min \sum_{i \in \{\text{supplier1},\ldots,\text{supplier8}\}} \sum_{j \in \{\text{demand1},\ldots,\text{demand8}\}} c_{ij} x_{ij}
$$

Subject to:

**1. Demand satisfaction for each customer group:**
\[
\begin{align*}
x_{\text{supplier1},\text{demand1}} + x_{\text{supplier2},\text{demand1}} + \cdots + x_{\text{supplier8},\text{demand1}} &= 9 \\
x_{\text{supplier1},\text{demand2}} + x_{\text{supplier2},\text{demand2}} + \cdots + x_{\text{supplier8},\text{demand2}} &= 66 \\
x_{\text{supplier1},\text{demand3}} + x_{\text{supplier2},\text{demand3}} + \cdots + x_{\text{supplier8},\text{demand3}} &= 56 \\
x_{\text{supplier1},\text{demand4}} + x_{\text{supplier2},\text{demand4}} + \cdots + x_{\text{supplier8},\text{demand4}} &= 17 \\
x_{\text{supplier1},\text{demand5}} + x_{\text{supplier2},\text{demand5}} + \cdots + x_{\text{supplier8},\text{demand5}} &= 43 \\
x_{\text{supplier1},\text{demand6}} + x_{\text{supplier2},\text{demand6}} + \cdots + x_{\text{supplier8},\text{demand6}} &= 62 \\
x_{\text{supplier1},\text{demand7}} + x_{\text{supplier2},\text{demand7}} + \cdots + x_{\text{supplier8},\text{demand7}} &= 10 \\
x_{\text{supplier1},\text{demand8}} + x_{\text{supplier2},\text{demand8}} + \cdots + x_{\text{supplier8},\text{demand8}} &= 37 \\
\end{align*}
\]

**2. Supply capacity for each supplier:**
\[
\begin{align*}
x_{\text{supplier1},\text{demand1}} + x_{\text{supplier1},\text{demand2}} + \cdots + x_{\text{supplier1},\text{demand8}} &\leq 60 \\
x_{\text{supplier2},\text{demand1}} + x_{\text{supplier2},\text{demand2}} + \cdots + x_{\text{supplier2},\text{demand8}} &\leq 22 \\
x_{\text{supplier3},\text{demand1}} + x_{\text{supplier3},\text{demand2}} + \cdots + x_{\text{supplier3},\text{demand8}} &\leq 16 \\
x_{\text{supplier4},\text{demand1}} + x_{\text{supplier4},\text{demand2}} + \cdots + x_{\text{supplier4},\text{demand8}} &\leq 14 \\
x_{\text{supplier5},\text{demand1}} + x_{\text{supplier5},\text{demand2}} + \cdots + x_{\text{supplier5},\text{demand8}} &\leq 19 \\
x_{\text{supplier6},\text{demand1}} + x_{\text{supplier6},\text{demand2}} + \cdots + x_{\text{supplier6},\text{demand8}} &\leq 70 \\
x_{\text{supplier7},\text{demand1}} + x_{\text{supplier7},\text{demand2}} + \cdots + x_{\text{supplier7},\text{demand8}} &\leq 60 \\
x_{\text{supplier8},\text{demand1}} + x_{\text{supplier8},\text{demand2}} + \cdots + x_{\text{supplier8},\text{demand8}} &\leq 39 \\
\end{align*}
\]

**3. Nonnegativity and integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{supplier1},\ldots,\text{supplier8}\},\ j \in \{\text{demand1},\ldots,\text{demand8}\}
\]

**Where:**

- $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$, as given in transportation_costs.csv:

|           | demand1 | demand2 | demand3 | demand4 | demand5 | demand6 | demand7 | demand8 |
|-----------|---------|---------|---------|---------|---------|---------|---------|---------|
| supply1   | 0.03020736643461065 | 229.50723504640203 | 198.62356558205792 | 12.995050640153751 | 211.20732124396406 | 134.9442985029274 | 9.822206398831067 | 11.394077543225675 |
| supply2   | 232.34691308087835 | 3.6258726438627473 | 0.28605434149404785 | 45.73127693242935 | 2.8304796563034573 | 107.05891033185472 | 299.96317913389305 | 23.79935436307657 |
| supply3   | 11.061938334356302 | 0.2041995326579051 | 0.2789447278030927 | 45.721912724349636 | 59.54895565737313 | 5.097536739581239 | 300.00118415135785 | 23.711282707746893 |
| supply4   | 235.1794835706472 | 43.794668963036194 | 40.709846782945924 | 0.07774496620087613 | 4.237728183419554 | 131.70915517494691 | 296.55587567706743 | 29.810940017561297 |
| supply5   | 211.85808746383796 | 47.60180876530328 | 50.04007716193931 | 86.14548807358399 | 0.06197897916874956 | 5.3345515296262205 | 270.06290423798396 | 3.853933133973331 |
| supply6   | 6.45506633554524 | 88.16323623354015 | 5.047091671641611 | 151.46120287365497 | 5.290760161059401 | 0.04602205335871525 | 9.93670660180487 | 103.75460989446313 |
| supply7   | 174.27229047340035 | 250.58223528739327 | 253.90413041857263 | 16.235467318386764 | 12.643140514778086 | 175.0672824108511 | 2.983839625303656 | 317.0655193866389 |
| supply8   | 207.87006253790491 | 1.517168471518212 | 24.027239288137153 | 27.133999276450346 | 73.20672468851855 | 125.72910359893308 | 15.463103251642147 | 0.20164987511903337 |

- Customer demands and supplier capacities are as follows:

| Customer   | Demand |
|------------|--------|
| demand1    | 9      |
| demand2    | 66     |
| demand3    | 56     |
| demand4    | 17     |
| demand5    | 43     |
| demand6    | 62     |
| demand7    | 10     |
| demand8    | 37     |

| Supplier   | Supply Capacity |
|------------|----------------|
| supplier1  | 60             |
| supplier2  | 22             |
| supplier3  | 16             |
| supplier4  | 14             |
| supplier5  | 19             |
| supplier6  | 70             |
| supplier7  | 60             |
| supplier8  | 39             |