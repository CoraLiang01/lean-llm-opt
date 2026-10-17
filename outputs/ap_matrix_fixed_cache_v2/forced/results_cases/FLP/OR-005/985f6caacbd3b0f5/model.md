##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous, nonnegative).

##### Parameters

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

- Customer group demands:
  - $d_{\text{demand1}} = 9$
  - $d_{\text{demand2}} = 66$
  - $d_{\text{demand3}} = 56$
  - $d_{\text{demand4}} = 17$
  - $d_{\text{demand5}} = 43$
  - $d_{\text{demand6}} = 62$
  - $d_{\text{demand7}} = 10$
  - $d_{\text{demand8}} = 37$

- Distribution center supply capacities:
  - $s_{\text{supplier1}} = 60$
  - $s_{\text{supplier2}} = 22$
  - $s_{\text{supplier3}} = 16$
  - $s_{\text{supplier4}} = 14$
  - $s_{\text{supplier5}} = 19$
  - $s_{\text{supplier6}} = 70$
  - $s_{\text{supplier7}} = 60$
  - $s_{\text{supplier8}} = 39$

- Transportation costs per unit ($c_{ij}$):

|            | demand1         | demand2         | demand3         | demand4         | demand5         | demand6         | demand7         | demand8         |
|------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| supply1    | 0.03020736643461065 | 229.50723504640203 | 198.62356558205792 | 12.995050640153751 | 211.20732124396406 | 134.9442985029274 | 9.822206398831067 | 11.394077543225675 |
| supply2    | 232.34691308087835 | 3.6258726438627473 | 0.28605434149404785 | 45.73127693242935 | 2.8304796563034573 | 107.05891033185472 | 299.96317913389305 | 23.79935436307657 |
| supply3    | 11.061938334356302 | 0.2041995326579051 | 0.2789447278030927 | 45.721912724349636 | 59.54895565737313 | 5.097536739581239 | 300.00118415135785 | 23.711282707746893 |
| supply4    | 235.1794835706472 | 43.794668963036194 | 40.709846782945924 | 0.07774496620087613 | 4.237728183419554 | 131.70915517494691 | 296.55587567706743 | 29.810940017561297 |
| supply5    | 211.85808746383796 | 47.60180876530328 | 50.04007716193931 | 86.14548807358399 | 0.06197897916874956 | 5.3345515296262205 | 270.06290423798396 | 3.853933133973331 |
| supply6    | 6.45506633554524 | 88.16323623354015 | 5.047091671641611 | 151.46120287365497 | 5.290760161059401 | 0.04602205335871525 | 9.93670660180487 | 103.75460989446313 |
| supply7    | 174.27229047340035 | 250.58223528739327 | 253.90413041857263 | 16.235467318386764 | 12.643140514778086 | 175.0672824108511 | 2.983839625303656 | 317.0655193866389 |
| supply8    | 207.87006253790491 | 1.517168471518212 | 24.027239288137153 | 27.133999276450346 | 73.20672468851855 | 125.72910359893308 | 15.463103251642147 | 0.20164987511903337 |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:** Each customer group receives exactly its demand.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supply capacity:** Each distribution center does not exceed its supply capacity.
   \[
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   \]

3. **Nonnegativity:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]

##### Retrieved Information

{
  "distribution_centers": [
    "supplier1",
    "supplier2",
    "supplier3",
    "supplier4",
    "supplier5",
    "supplier6",
    "supplier7",
    "supplier8"
  ],
  "customer_groups": [
    "demand1",
    "demand2",
    "demand3",
    "demand4",
    "demand5",
    "demand6",
    "demand7",
    "demand8"
  ],
  "demand": {
    "demand1": 9,
    "demand2": 66,
    "demand3": 56,
    "demand4": 17,
    "demand5": 43,
    "demand6": 62,
    "demand7": 10,
    "demand8": 37
  },
  "supply_capacity": {
    "supplier1": 60,
    "supplier2": 22,
    "supplier3": 16,
    "supplier4": 14,
    "supplier5": 19,
    "supplier6": 70,
    "supplier7": 60,
    "supplier8": 39
  },
  "cost": {
    "supply1": {
      "demand1": 0.03020736643461065,
      "demand2": 229.50723504640203,
      "demand3": 198.62356558205792,
      "demand4": 12.995050640153751,
      "demand5": 211.20732124396406,
      "demand6": 134.9442985029274,
      "demand7": 9.822206398831067,
      "demand8": 11.394077543225675
    },
    "supply2": {
      "demand1": 232.34691308087835,
      "demand2": 3.6258726438627473,
      "demand3": 0.28605434149404785,
      "demand4": 45.73127693242935,
      "demand5": 2.8304796563034573,
      "demand6": 107.05891033185472,
      "demand7": 299.96317913389305,
      "demand8": 23.79935436307657
    },
    "supply3": {
      "demand1": 11.061938334356302,
      "demand2": 0.2041995326579051,
      "demand3": 0.2789447278030927,
      "demand4": 45.721912724349636,
      "demand5": 59.54895565737313,
      "demand6": 5.097536739581239,
      "demand7": 300.00118415135785,
      "demand8": 23.711282707746893
    },
    "supply4": {
      "demand1": 235.1794835706472,
      "demand2": 43.794668963036194,
      "demand3": 40.709846782945924,
      "demand4": 0.07774496620087613,
      "demand5": 4.237728183419554,
      "demand6": 131.70915517494691,
      "demand7": 296.55587567706743,
      "demand8": 29.810940017561297
    },
    "supply5": {
      "demand1": 211.85808746383796,
      "demand2": 47.60180876530328,
      "demand3": 50.04007716193931,
      "demand4": 86.14548807358399,
      "demand5": 0.06197897916874956,
      "demand6": 5.3345515296262205,
      "demand7": 270.06290423798396,
      "demand8": 3.853933133973331
    },
    "supply6": {
      "demand1": 6.45506633554524,
      "demand2": 88.16323623354015,
      "demand3": 5.047091671641611,
      "demand4": 151.46120287365497,
      "demand5": 5.290760161059401,
      "demand6": 0.04602205335871525,
      "demand7": 9.93670660180487,
      "demand8": 103.75460989446313
    },
    "supply7": {
      "demand1": 174.27229047340035,
      "demand2": 250.58223528739327,
      "demand3": 253.90413041857263,
      "demand4": 16.235467318386764,
      "demand5": 12.643140514778086,
      "demand6": 175.0672824108511,
      "demand7": 2.983839625303656,
      "demand8": 317.0655193866389
    },
    "supply8": {
      "demand1": 207.87006253790491,
      "demand2": 1.517168471518212,
      "demand3": 24.027239288137153,
      "demand4": 27.133999276450346,
      "demand5": 73.20672468851855,
      "demand6": 125.72910359893308,
      "demand7": 15.463103251642147,
      "demand8": 0.20164987511903337
    }
  }
}