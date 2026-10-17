Sets:
- Let S = {supplier1, supplier2, supplier3, supplier4, supplier5, supplier6, supplier7, supplier8} (distribution centers)
- Let D = {demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8} (customer groups)

Parameters (from CSVs, in source order):

Customer Demands (from customer_demand.csv):
- demand1: 9
- demand2: 66
- demand3: 56
- demand4: 17
- demand5: 43
- demand6: 62
- demand7: 10
- demand8: 37

Supply Capacities (from supply_capacity.csv):
- supplier1: 60
- supplier2: 22
- supplier3: 16
- supplier4: 14
- supplier5: 19
- supplier6: 70
- supplier7: 60
- supplier8: 39

Transportation Costs (from transportation_costs.csv, in source order):

|            | demand1           | demand2           | demand3           | demand4           | demand5           | demand6           | demand7           | demand8           |
|------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| supplier1  | 0.03020736643461065 | 229.50723504640203 | 198.62356558205792 | 12.995050640153751 | 211.20732124396406 | 134.9442985029274 | 9.822206398831067 | 11.394077543225675 |
| supplier2  | 232.34691308087835 | 3.6258726438627473 | 0.28605434149404785 | 45.73127693242935 | 2.8304796563034573 | 107.05891033185472 | 299.96317913389305 | 23.79935436307657 |
| supplier3  | 11.061938334356302 | 0.2041995326579051 | 0.2789447278030927 | 45.721912724349636 | 59.54895565737313 | 5.097536739581239 | 300.00118415135785 | 23.711282707746893 |
| supplier4  | 235.1794835706472 | 43.794668963036194 | 40.709846782945924 | 0.07774496620087613 | 4.237728183419554 | 131.70915517494691 | 296.55587567706743 | 29.810940017561297 |
| supplier5  | 211.85808746383796 | 47.60180876530328 | 50.04007716193931 | 86.14548807358399 | 0.06197897916874956 | 5.3345515296262205 | 270.06290423798396 | 3.853933133973331 |
| supplier6  | 6.45506633554524 | 88.16323623354015 | 5.047091671641611 | 151.46120287365497 | 5.290760161059401 | 0.04602205335871525 | 9.93670660180487 | 103.75460989446313 |
| supplier7  | 174.27229047340035 | 250.58223528739327 | 253.90413041857263 | 16.235467318386764 | 12.643140514778086 | 175.0672824108511 | 2.983839625303656 | 317.0655193866389 |
| supplier8  | 207.87006253790491 | 1.517168471518212 | 24.027239288137153 | 27.133999276450346 | 73.20672468851855 | 125.72910359893308 | 15.463103251642147 | 0.20164987511903337 |

Decision Variables:
- Let x_{s,d} ≥ 0: quantity of goods shipped from supplier s ∈ S to customer group d ∈ D

Objective:
Minimize total transportation cost:
minimize
  0.03020736643461065 x_{supplier1,demand1} + 229.50723504640203 x_{supplier1,demand2} + 198.62356558205792 x_{supplier1,demand3} + 12.995050640153751 x_{supplier1,demand4} + 211.20732124396406 x_{supplier1,demand5} + 134.9442985029274 x_{supplier1,demand6} + 9.822206398831067 x_{supplier1,demand7} + 11.394077543225675 x_{supplier1,demand8}
+ 232.34691308087835 x_{supplier2,demand1} + 3.6258726438627473 x_{supplier2,demand2} + 0.28605434149404785 x_{supplier2,demand3} + 45.73127693242935 x_{supplier2,demand4} + 2.8304796563034573 x_{supplier2,demand5} + 107.05891033185472 x_{supplier2,demand6} + 299.96317913389305 x_{supplier2,demand7} + 23.79935436307657 x_{supplier2,demand8}
+ 11.061938334356302 x_{supplier3,demand1} + 0.2041995326579051 x_{supplier3,demand2} + 0.2789447278030927 x_{supplier3,demand3} + 45.721912724349636 x_{supplier3,demand4} + 59.54895565737313 x_{supplier3,demand5} + 5.097536739581239 x_{supplier3,demand6} + 300.00118415135785 x_{supplier3,demand7} + 23.711282707746893 x_{supplier3,demand8}
+ 235.1794835706472 x_{supplier4,demand1} + 43.794668963036194 x_{supplier4,demand2} + 40.709846782945924 x_{supplier4,demand3} + 0.07774496620087613 x_{supplier4,demand4} + 4.237728183419554 x_{supplier4,demand5} + 131.70915517494691 x_{supplier4,demand6} + 296.55587567706743 x_{supplier4,demand7} + 29.810940017561297 x_{supplier4,demand8}
+ 211.85808746383796 x_{supplier5,demand1} + 47.60180876530328 x_{supplier5,demand2} + 50.04007716193931 x_{supplier5,demand3} + 86.14548807358399 x_{supplier5,demand4} + 0.06197897916874956 x_{supplier5,demand5} + 5.3345515296262205 x_{supplier5,demand6} + 270.06290423798396 x_{supplier5,demand7} + 3.853933133973331 x_{supplier5,demand8}
+ 6.45506633554524 x_{supplier6,demand1} + 88.16323623354015 x_{supplier6,demand2} + 5.047091671641611 x_{supplier6,demand3} + 151.46120287365497 x_{supplier6,demand4} + 5.290760161059401 x_{supplier6,demand5} + 0.04602205335871525 x_{supplier6,demand6} + 9.93670660180487 x_{supplier6,demand7} + 103.75460989446313 x_{supplier6,demand8}
+ 174.27229047340035 x_{supplier7,demand1} + 250.58223528739327 x_{supplier7,demand2} + 253.90413041857263 x_{supplier7,demand3} + 16.235467318386764 x_{supplier7,demand4} + 12.643140514778086 x_{supplier7,demand5} + 175.0672824108511 x_{supplier7,demand6} + 2.983839625303656 x_{supplier7,demand7} + 317.0655193866389 x_{supplier7,demand8}
+ 207.87006253790491 x_{supplier8,demand1} + 1.517168471518212 x_{supplier8,demand2} + 24.027239288137153 x_{supplier8,demand3} + 27.133999276450346 x_{supplier8,demand4} + 73.20672468851855 x_{supplier8,demand5} + 125.72910359893308 x_{supplier8,demand6} + 15.463103251642147 x_{supplier8,demand7} + 0.20164987511903337 x_{supplier8,demand8}

Subject to:

1. Demand satisfaction (for each customer group d ∈ D):
   For each d ∈ D,
   ∑_{s ∈ S} x_{s,d} = demand_d

   - ∑_{s ∈ S} x_{s,demand1} = 9
   - ∑_{s ∈ S} x_{s,demand2} = 66
   - ∑_{s ∈ S} x_{s,demand3} = 56
   - ∑_{s ∈ S} x_{s,demand4} = 17
   - ∑_{s ∈ S} x_{s,demand5} = 43
   - ∑_{s ∈ S} x_{s,demand6} = 62
   - ∑_{s ∈ S} x_{s,demand7} = 10
   - ∑_{s ∈ S} x_{s,demand8} = 37

2. Supply capacity (for each supplier s ∈ S):
   For each s ∈ S,
   ∑_{d ∈ D} x_{s,d} ≤ supply_capacity_s

   - ∑_{d ∈ D} x_{supplier1,d} ≤ 60
   - ∑_{d ∈ D} x_{supplier2,d} ≤ 22
   - ∑_{d ∈ D} x_{supplier3,d} ≤ 16
   - ∑_{d ∈ D} x_{supplier4,d} ≤ 14
   - ∑_{d ∈ D} x_{supplier5,d} ≤ 19
   - ∑_{d ∈ D} x_{supplier6,d} ≤ 70
   - ∑_{d ∈ D} x_{supplier7,d} ≤ 60
   - ∑_{d ∈ D} x_{supplier8,d} ≤ 39

3. Nonnegativity:
   For all s ∈ S, d ∈ D:
   x_{s,d} ≥ 0

This is a complete numerical linear programming formulation for the described transportation problem, using all identifiers and coefficients from the provided CSVs, preserving source order and all original semantics.