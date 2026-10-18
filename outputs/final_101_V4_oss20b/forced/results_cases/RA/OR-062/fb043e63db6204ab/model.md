##### Objective Function: 

$ \quad \min \sum_{i} \text{fixed\_costs}_{i} \cdot y_{i} \;+\; \sum_{i}\sum_{j} \text{transportation\_costs}_{ij} \cdot x_{ij} $

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$ \quad \sum_{i} x_{ij} \;\geq\; \text{demand}_{j}, \quad \forall j $

###### 2. Supplier Activation Constraints: 

$ \quad x_{ij} \;\leq\; \text{demand}_{j} \cdot y_{i}, \quad \forall i,\;\forall j $

###### 3. Non‑negativity and Integrality Constraints: 

$ \quad x_{ij} \;\in\; \mathbb{Z}_{\geq 0}, \quad \forall i,\;\forall j $  

$ \quad y_{i} \;\in\; \{0,1\}, \quad \forall i $

###### 4. Variable Indexing: 

- $i$ indexes the suppliers: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES.  
- $j$ indexes the customers: Customer_1, Customer_2, Customer_3, Customer_4, Customer_5.  

###### 5. Parameter Definitions:  

- $\text{fixed\_costs}_{i}$ is the fixed cost for opening supplier $i$.  
- $\text{transportation\_costs}_{ij}$ is the per‑unit transportation cost from supplier $i$ to customer $j$.  
- $\text{demand}_{j}$ is the total demand of customer $j$.

###### 6. Parameter Values (Retrieved Information)

{
  "suppliers": [
    "MOUNT AYR",
    "WAUKEE",
    "WAVERLY",
    "PELLA",
    "DES MOINES"
  ],
  "fixed_costs": {
    "MOUNT AYR": "96.58",
    "WAUKEE": "94.06",
    "WAVERLY": "94.37",
    "PELLA": "82.88",
    "DES MOINES": "94.95999999999999"
  },
  "customers": [
    "Customer_1",
    "Customer_2",
    "Customer_3",
    "Customer_4",
    "Customer_5"
  ],
  "demand": {
    "Customer_1": "2397",
    "Customer_2": "1889",
    "Customer_3": "2518",
    "Customer_4": "3218",
    "Customer_5": "1813"
  },
  "transportation_costs": {
    "MOUNT AYR": {
      "CLARINDA": "694.6799999999999",
      "FORT MADISON": "17.48",
      "SIOUX CITY": "20.07",
      "TOLEDO": "199.02",
      "BANCROFT": "1685.53"
    },
    "WAUKEE": {
      "CLARINDA": "15.13",
      "FORT MADISON": "1.5",
      "SIOUX CITY": "1.43",
      "TOLEDO": "27.88",
      "BANCROFT": "90.69"
    },
    "WAVERLY": {
      "CLARINDA": "2.34",
      "FORT MADISON": "349.34",
      "SIOUX CITY": "246.6",
      "TOLEDO": "41.3",
      "BANCROFT": "78.73"
    },
    "PELLA": {
      "CLARINDA": "1181.6",
      "FORT MADISON": "1458.53",
      "SIOUX CITY": "1646.36",
      "TOLEDO": "1924.55",
      "BANCROFT": "38.93"
    },
    "DES MOINES": {
      "CLARINDA": "1030.8",
      "FORT MADISON": "43.48",
      "SIOUX CITY": "932.4299999999999",
      "TOLEDO": "55.39",
      "BANCROFT": "103.84"
    }
  }
}