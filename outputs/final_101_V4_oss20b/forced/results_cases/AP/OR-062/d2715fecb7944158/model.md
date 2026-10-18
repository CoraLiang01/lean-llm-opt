##### Objective Function: 

$ \min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij} $

##### Constraints

###### 1. Demand Satisfaction: 

$ \sum_{i \in S} x_{ij} = d_j \quad \forall j \in C $

###### 2. Supplier Activation Linking: 

$ x_{ij} \leq M_j y_i \quad \forall i \in S, j \in C $

###### 3. Non‑negativity: 

$ x_{ij} \ge 0 \quad \forall i \in S, j \in C $

###### 4. Binary Variables: 

$ y_i \in \{0,1\} \quad \forall i \in S $

##### Retrieved Information

{
  "fixed_costs": {
    "MOUNT AYR": 96.58,
    "WAUKEE": 94.06,
    "WAVERLY": 94.37,
    "PELLA": 82.88,
    "DES MOINES": 94.96
  },
  "transportation_costs": {
    "MOUNT AYR": {
      "BANCROFT": 1685.53,
      "CLARINDA": 694.68,
      "FORT MADISON": 17.48,
      "SIOUX CITY": 20.07,
      "TOLEDO": 199.02
    },
    "WAUKEE": {
      "BANCROFT": 90.69,
      "CLARINDA": 15.13,
      "FORT MADISON": 1.5,
      "SIOUX CITY": 1.43,
      "TOLEDO": 27.88
    },
    "WAVERLY": {
      "BANCROFT": 78.73,
      "CLARINDA": 2.34,
      "FORT MADISON": 349.34,
      "SIOUX CITY": 246.6,
      "TOLEDO": 41.3
    },
    "PELLA": {
      "BANCROFT": 38.93,
      "CLARINDA": 1181.6,
      "FORT MADISON": 1458.53,
      "SIOUX CITY": 1646.36,
      "TOLEDO": 1924.55
    },
    "DES MOINES": {
      "BANCROFT": 103.84,
      "CLARINDA": 1030.8,
      "FORT MADISON": 43.48,
      "SIOUX CITY": 932.43,
      "TOLEDO": 55.39
    }
  },
  "demand": {
    "Customer_1": 2397,
    "Customer_2": 1889,
    "Customer_3": 2518,
    "Customer_4": 3218,
    "Customer_5": 1813
  },
  "suppliers": [
    "MOUNT AYR",
    "WAUKEE",
    "WAVERLY",
    "PELLA",
    "DES MOINES"
  ],
  "stores": [
    "BANCROFT",
    "CLARINDA",
    "FORT MADISON",
    "SIOUX CITY",
    "TOLEDO"
  ]
}