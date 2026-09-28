##### Decision Variables

Let $x_i$ denote the number of units of Fashion product $i$ to be fulfilled, for each product $i$ in the set of Fashion products.

##### Parameters

Let $P$ be the set of Fashion products:

$P = \{$
Fashion accessories\_10.18,
Fashion accessories\_12.09,
Fashion accessories\_12.19,
Fashion accessories\_12.54,
Fashion accessories\_12.78,
Fashion accessories\_14.48,
Fashion accessories\_15.43,
Fashion accessories\_15.5,
Fashion accessories\_15.62,
Fashion accessories\_16.28,
Fashion accessories\_16.45,
Fashion accessories\_17.48,
Fashion accessories\_17.49,
Fashion accessories\_17.87,
Fashion accessories\_17.94,
Fashion accessories\_18.08,
Fashion accessories\_19.66,
Fashion accessories\_19.7,
Fashion accessories\_19.77,
Fashion accessories\_20.01,
Fashion accessories\_21.32,
Fashion accessories\_21.48,
Fashion accessories\_21.94,
Fashion accessories\_22.32,
Fashion accessories\_22.51,
Fashion accessories\_23.82,
Fashion accessories\_25.42,
Fashion accessories\_25.56,
Fashion accessories\_27.02,
Fashion accessories\_27.18,
Fashion accessories\_27.38,
Fashion accessories\_29.42,
Fashion accessories\_29.56,
Fashion accessories\_30.14,
Fashion accessories\_30.37,
Fashion accessories\_30.61,
Fashion accessories\_30.62,
Fashion accessories\_31.73,
Fashion accessories\_31.9,
Fashion accessories\_32.62,
Fashion accessories\_33.52,
Fashion accessories\_33.63,
Fashion accessories\_34.7,
Fashion accessories\_35.19,
Fashion accessories\_36.51,
Fashion accessories\_36.85,
Fashion accessories\_37.15,
Fashion accessories\_37.55,
Fashion accessories\_37.95
$\}$

For each product $i \in P$:

- $r_i$ = Revenue per unit of product $i$
- $s_i$ = Initial Inventory of product $i$
- $d_i$ = Demand for product $i$

The parameter values are:

| Product Name                  | $r_i$  | $s_i$ | $d_i$ |
|-------------------------------|--------|-------|-------|
| Fashion accessories_10.18     | 10.18  | 80    | 12    |
| Fashion accessories_12.09     | 12.09  | 10    | 2     |
| Fashion accessories_12.19     | 12.19  | 80    | 10    |
| Fashion accessories_12.54     | 12.54  | 10    | 2     |
| Fashion accessories_12.78     | 12.78  | 10    | 2     |
| Fashion accessories_14.48     | 14.48  | 40    | 5     |
| Fashion accessories_15.43     | 15.43  | 10    | 2     |
| Fashion accessories_15.5      | 15.5   | 10    | 2     |
| Fashion accessories_15.62     | 15.62  | 80    | 11    |
| Fashion accessories_16.28     | 16.28  | 10    | 2     |
| Fashion accessories_16.45     | 16.45  | 40    | 6     |
| Fashion accessories_17.48     | 17.48  | 60    | 9     |
| Fashion accessories_17.49     | 17.49  | 100   | 14    |
| Fashion accessories_17.87     | 17.87  | 40    | 6     |
| Fashion accessories_17.94     | 17.94  | 50    | 8     |
| Fashion accessories_18.08     | 18.08  | 40    | 6     |
| Fashion accessories_19.66     | 19.66  | 100   | 14    |
| Fashion accessories_19.7      | 19.7   | 10    | 2     |
| Fashion accessories_19.77     | 19.77  | 100   | 14    |
| Fashion accessories_20.01     | 20.01  | 90    | 11    |
| Fashion accessories_21.32     | 21.32  | 10    | 2     |
| Fashion accessories_21.48     | 21.48  | 20    | 3     |
| Fashion accessories_21.94     | 21.94  | 50    | 8     |
| Fashion accessories_22.32     | 22.32  | 80    | 10    |
| Fashion accessories_22.51     | 22.51  | 70    | 10    |
| Fashion accessories_23.82     | 23.82  | 50    | 7     |
| Fashion accessories_25.42     | 25.42  | 80    | 11    |
| Fashion accessories_25.56     | 25.56  | 70    | 11    |
| Fashion accessories_27.02     | 27.02  | 30    | 5     |
| Fashion accessories_27.18     | 27.18  | 20    | 3     |
| Fashion accessories_27.38     | 27.38  | 60    | 8     |
| Fashion accessories_29.42     | 29.42  | 100   | 13    |
| Fashion accessories_29.56     | 29.56  | 50    | 7     |
| Fashion accessories_30.14     | 30.14  | 100   | 14    |
| Fashion accessories_30.37     | 30.37  | 30    | 4     |
| Fashion accessories_30.61     | 30.61  | 10    | 2     |
| Fashion accessories_30.62     | 30.62  | 10    | 2     |
| Fashion accessories_31.73     | 31.73  | 90    | 14    |
| Fashion accessories_31.9      | 31.9   | 10    | 2     |
| Fashion accessories_32.62     | 32.62  | 40    | 6     |
| Fashion accessories_33.52     | 33.52  | 10    | 2     |
| Fashion accessories_33.63     | 33.63  | 10    | 2     |
| Fashion accessories_34.7      | 34.7   | 20    | 3     |
| Fashion accessories_35.19     | 35.19  | 100   | 14    |
| Fashion accessories_36.51     | 36.51  | 90    | 12    |
| Fashion accessories_36.85     | 36.85  | 50    | 7     |
| Fashion accessories_37.15     | 37.15  | 40    | 6     |
| Fashion accessories_37.55     | 37.55  | 100   | 13    |
| Fashion accessories_37.95     | 37.95  | 100   | 14    |

##### Objective Function

$\max \sum_{i \in P} r_i x_i$

##### Constraints

For each product $i \in P$:

- Inventory constraint: $x_i \leq s_i$
- Demand constraint: $x_i \leq d_i$
- Non-negativity and integrality: $x_i \geq 0$, $x_i \in \mathbb{Z}$

##### Complete Mathematical Model

$\max \sum_{i \in P} r_i x_i$

subject to

$\quad x_i \leq s_i \quad \forall i \in P$

$\quad x_i \leq d_i \quad \forall i \in P$

$\quad x_i \geq 0,\ x_i \in \mathbb{Z} \quad \forall i \in P$

##### Retrieved Information

{
  "products": [
    "Fashion accessories_10.18",
    "Fashion accessories_12.09",
    "Fashion accessories_12.19",
    "Fashion accessories_12.54",
    "Fashion accessories_12.78",
    "Fashion accessories_14.48",
    "Fashion accessories_15.43",
    "Fashion accessories_15.5",
    "Fashion accessories_15.62",
    "Fashion accessories_16.28",
    "Fashion accessories_16.45",
    "Fashion accessories_17.48",
    "Fashion accessories_17.49",
    "Fashion accessories_17.87",
    "Fashion accessories_17.94",
    "Fashion accessories_18.08",
    "Fashion accessories_19.66",
    "Fashion accessories_19.7",
    "Fashion accessories_19.77",
    "Fashion accessories_20.01",
    "Fashion accessories_21.32",
    "Fashion accessories_21.48",
    "Fashion accessories_21.94",
    "Fashion accessories_22.32",
    "Fashion accessories_22.51",
    "Fashion accessories_23.82",
    "Fashion accessories_25.42",
    "Fashion accessories_25.56",
    "Fashion accessories_27.02",
    "Fashion accessories_27.18",
    "Fashion accessories_27.38",
    "Fashion accessories_29.42",
    "Fashion accessories_29.56",
    "Fashion accessories_30.14",
    "Fashion accessories_30.37",
    "Fashion accessories_30.61",
    "Fashion accessories_30.62",
    "Fashion accessories_31.73",
    "Fashion accessories_31.9",
    "Fashion accessories_32.62",
    "Fashion accessories_33.52",
    "Fashion accessories_33.63",
    "Fashion accessories_34.7",
    "Fashion accessories_35.19",
    "Fashion accessories_36.51",
    "Fashion accessories_36.85",
    "Fashion accessories_37.15",
    "Fashion accessories_37.55",
    "Fashion accessories_37.95"
  ],
  "revenue": {
    "Fashion accessories_10.18": 10.18,
    "Fashion accessories_12.09": 12.09,
    "Fashion accessories_12.19": 12.19,
    "Fashion accessories_12.54": 12.54,
    "Fashion accessories_12.78": 12.78,
    "Fashion accessories_14.48": 14.48,
    "Fashion accessories_15.43": 15.43,
    "Fashion accessories_15.5": 15.5,
    "Fashion accessories_15.62": 15.62,
    "Fashion accessories_16.28": 16.28,
    "Fashion accessories_16.45": 16.45,
    "Fashion accessories_17.48": 17.48,
    "Fashion accessories_17.49": 17.49,
    "Fashion accessories_17.87": 17.87,
    "Fashion accessories_17.94": 17.94,
    "Fashion accessories_18.08": 18.08,
    "Fashion accessories_19.66": 19.66,
    "Fashion accessories_19.7": 19.7,
    "Fashion accessories_19.77": 19.77,
    "Fashion accessories_20.01": 20.01,
    "Fashion accessories_21.32": 21.32,
    "Fashion accessories_21.48": 21.48,
    "Fashion accessories_21.94": 21.94,
    "Fashion accessories_22.32": 22.32,
    "Fashion accessories_22.51": 22.51,
    "Fashion accessories_23.82": 23.82,
    "Fashion accessories_25.42": 25.42,
    "Fashion accessories_25.56": 25.56,
    "Fashion accessories_27.02": 27.02,
    "Fashion accessories_27.18": 27.18,
    "Fashion accessories_27.38": 27.38,
    "Fashion accessories_29.42": 29.42,
    "Fashion accessories_29.56": 29.56,
    "Fashion accessories_30.14": 30.14,
    "Fashion accessories_30.37": 30.37,
    "Fashion accessories_30.61": 30.61,
    "Fashion accessories_30.62": 30.62,
    "Fashion accessories_31.73": 31.73,
    "Fashion accessories_31.9": 31.9,
    "Fashion accessories_32.62": 32.62,
    "Fashion accessories_33.52": 33.52,
    "Fashion accessories_33.63": 33.63,
    "Fashion accessories_34.7": 34.7,
    "Fashion accessories_35.19": 35.19,
    "Fashion accessories_36.51": 36.51,
    "Fashion accessories_36.85": 36.85,
    "Fashion accessories_37.15": 37.15,
    "Fashion accessories_37.55": 37.55,
    "Fashion accessories_37.95": 37.95
  },
  "initial_inventory": {
    "Fashion accessories_10.18": 80,
    "Fashion accessories_12.09": 10,
    "Fashion accessories_12.19": 80,
    "Fashion accessories_12.54": 10,
    "Fashion accessories_12.78": 10,
    "Fashion accessories_14.48": 40,
    "Fashion accessories_15.43": 10,
    "Fashion accessories_15.5": 10,
    "Fashion accessories_15.62": 80,
    "Fashion accessories_16.28": 10,
    "Fashion accessories_16.45": 40,
    "Fashion accessories_17.48": 60,
    "Fashion accessories_17.49": 100,
    "Fashion accessories_17.87": 40,
    "Fashion accessories_17.94": 50,
    "Fashion accessories_18.08": 40,
    "Fashion accessories_19.66": 100,
    "Fashion accessories_19.7": 10,
    "Fashion accessories_19.77": 100,
    "Fashion accessories_20.01": 90,
    "Fashion accessories_21.32": 10,
    "Fashion accessories_21.48": 20,
    "Fashion accessories_21.94": 50,
    "Fashion accessories_22.32": 80,
    "Fashion accessories_22.51": 70,
    "Fashion accessories_23.82": 50,
    "Fashion accessories_25.42": 80,
    "Fashion accessories_25.56": 70,
    "Fashion accessories_27.02": 30,
    "Fashion accessories_27.18": 20,
    "Fashion accessories_27.38": 60,
    "Fashion accessories_29.42": 100,
    "Fashion accessories_29.56": 50,
    "Fashion accessories_30.14": 100,
    "Fashion accessories_30.37": 30,
    "Fashion accessories_30.61": 10,
    "Fashion accessories_30.62": 10,
    "Fashion accessories_31.73": 90,
    "Fashion accessories_31.9": 10,
    "Fashion accessories_32.62": 40,
    "Fashion accessories_33.52": 10,
    "Fashion accessories_33.63": 10,
    "Fashion accessories_34.7": 20,
    "Fashion accessories_35.19": 100,
    "Fashion accessories_36.51": 90,
    "Fashion accessories_36.85": 50,
    "Fashion accessories_37.15": 40,
    "Fashion accessories_37.55": 100,
    "Fashion accessories_37.95": 100
  },
  "demand": {
    "Fashion accessories_10.18": 12,
    "Fashion accessories_12.09": 2,
    "Fashion accessories_12.19": 10,
    "Fashion accessories_12.54": 2,
    "Fashion accessories_12.78": 2,
    "Fashion accessories_14.48": 5,
    "Fashion accessories_15.43": 2,
    "Fashion accessories_15.5": 2,
    "Fashion accessories_15.62": 11,
    "Fashion accessories_16.28": 2,
    "Fashion accessories_16.45": 6,
    "Fashion accessories_17.48": 9,
    "Fashion accessories_17.49": 14,
    "Fashion accessories_17.87": 6,
    "Fashion accessories_17.94": 8,
    "Fashion accessories_18.08": 6,
    "Fashion accessories_19.66": 14,
    "Fashion accessories_19.7": 2,
    "Fashion accessories_19.77": 14,
    "Fashion accessories_20.01": 11,
    "Fashion accessories_21.32": 2,
    "Fashion accessories_21.48": 3,
    "Fashion accessories_21.94": 8,
    "Fashion accessories_22.32": 10,
    "Fashion accessories_22.51": 10,
    "Fashion accessories_23.82": 7,
    "Fashion accessories_25.42": 11,
    "Fashion accessories_25.56": 11,
    "Fashion accessories_27.02": 5,
    "Fashion accessories_27.18": 3,
    "Fashion accessories_27.38": 8,
    "Fashion accessories_29.42": 13,
    "Fashion accessories_29.56": 7,
    "Fashion accessories_30.14": 14,
    "Fashion accessories_30.37": 4,
    "Fashion accessories_30.61": 2,
    "Fashion accessories_30.62": 2,
    "Fashion accessories_31.73": 14,
    "Fashion accessories_31.9": 2,
    "Fashion accessories_32.62": 6,
    "Fashion accessories_33.52": 2,
    "Fashion accessories_33.63": 2,
    "Fashion accessories_34.7": 3,
    "Fashion accessories_35.19": 14,
    "Fashion accessories_36.51": 12,
    "Fashion accessories_36.85": 7,
    "Fashion accessories_37.15": 6,
    "Fashion accessories_37.55": 13,
    "Fashion accessories_37.95": 14
  }
}