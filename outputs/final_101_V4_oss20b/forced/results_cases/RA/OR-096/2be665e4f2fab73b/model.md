##### Objective Function: 

$ \quad \quad \min \sum_{i} \sum_{j} d_{ij} \cdot (x_{ij}^w + x_{ij}^n)$

##### Constraints

###### 1. Capacity Constraints: 

$ \quad \quad \sum_{i} (x_{ij}^w + x_{ij}^n) \leq C_j, \quad \forall j$

###### 2. Population Constraints: 

$ \quad \quad \sum_{j} x_{ij}^w = W_i, \quad \forall i$

$ \quad \quad \sum_{j} x_{ij}^n = N_i, \quad \forall i$

###### 3. Racial Balance Constraints: 

$ \quad \quad \sum_{i} x_{ij}^w \leq 0.7 \sum_{i} (x_{ij}^w + x_{ij}^n), \quad \forall j$

$ \quad \quad \sum_{i} x_{ij}^w \geq 0.5 \sum_{i} (x_{ij}^w + x_{ij}^n), \quad \forall j$

###### 4. Non-negativity and Integrality Constraints: 

$ \quad \quad x_{ij}^w, x_{ij}^n \in \mathbb{Z}_{\geq 0}, \quad \forall i,j$

###### Retrieved Information

{
  "schools": [
    {"School": "I", "Capacity": "2028"},
    {"School": "II", "Capacity": "1560"}
  ],
  "neighborhoods": [
    {"Neighborhood": "N01", "Population_White": "78", "Population_NonWhite": "22"},
    {"Neighborhood": "N02", "Population_White": "57", "Population_NonWhite": "33"},
    {"Neighborhood": "N03", "Population_White": "47", "Population_NonWhite": "63"},
    {"Neighborhood": "N04", "Population_White": "78", "Population_NonWhite": "22"},
    {"Neighborhood": "N05", "Population_White": "57", "Population_NonWhite": "33"},
    {"Neighborhood": "N06", "Population_White": "47", "Population_NonWhite": "63"},
    {"Neighborhood": "N07", "Population_White": "78", "Population_NonWhite": "22"},
    {"Neighborhood": "N08", "Population_White": "57", "Population_NonWhite": "33"},
    {"Neighborhood": "N09", "Population_White": "46", "Population_NonWhite": "64"},
    {"Neighborhood": "N10", "Population_White": "77", "Population_NonWhite": "23"},
    {"Neighborhood": "N11", "Population_White": "56", "Population_NonWhite": "34"},
    {"Neighborhood": "N12", "Population_White": "46", "Population_NonWhite": "64"},
    {"Neighborhood": "N13", "Population_White": "77", "Population_NonWhite": "23"},
    {"Neighborhood": "N14", "Population_White": "56", "Population_NonWhite": "34"},
    {"Neighborhood": "N15", "Population_White": "46", "Population_NonWhite": "64"},
    {"Neighborhood": "N16", "Population_White": "77", "Population_NonWhite": "23"},
    {"Neighborhood": "N17", "Population_White": "56", "Population_NonWhite": "34"},
    {"Neighborhood": "N18", "Population_White": "46", "Population_NonWhite": "64"},
    {"Neighborhood": "N19", "Population_White": "77", "Population_NonWhite": "23"},
    {"Neighborhood": "N20", "Population_White": "56", "Population_NonWhite": "34"},
    {"Neighborhood": "N21", "Population_White": "46", "Population_NonWhite": "64"},
    {"Neighborhood": "N22", "Population_White": "77", "Population_NonWhite": "23"},
    {"Neighborhood": "N23", "Population_White": "56", "Population_NonWhite": "34"},
    {"Neighborhood": "N24", "Population_White": "46", "Population_NonWhite": "64"},
    {"Neighborhood": "N25", "Population_White": "77", "Population_NonWhite": "23"},
    {"Neighborhood": "N26", "Population_White": "56", "Population_NonWhite": "34"},
    {"Neighborhood": "N27", "Population_White": "46", "Population_NonWhite": "64"},
    {"Neighborhood": "N28", "Population_White": "77", "Population_NonWhite": "23"},
    {"Neighborhood": "N29", "Population_White": "56", "Population_NonWhite": "34"},
    {"Neighborhood": "N30", "Population_White": "46", "Population_NonWhite": "64"},
    {"Neighborhood": "N31", "Population_White": "74", "Population_NonWhite": "46"}
  ],
  "distances": [
    {"School": "I", "N01": "1.25", "N02": "1.3", "N03": "1.35", "N04": "1.4", "N05": "1.45", "N06": "1.5", "N07": "1.55", "N08": "1.6", "N09": "1.65", "N10": "1.7", "N11": "1.75", "N12": "1.8", "N13": "1.85", "N14": "1.9", "N15": "1.95", "N16": "2.0", "N17": "3.08", "N18": "3.16", "N19": "3.24", "N20": "3.32", "N21": "3.4", "N22": "3.48", "N23": "3.56", "N24": "3.64", "N25": "3.7199999999999998", "N26": "3.8", "N27": "3.88", "N28": "3.96", "N29": "4.04", "N30": "4.12", "N31": "4.2"},
    {"School": "II", "N01": "3.08", "N02": "3.16", "N03": "3.24", "N04": "3.32", "N05": "3.4", "N06": "3.48", "N07": "3.56", "N08": "3.64", "N09": "3.7199999999999998", "N10": "3.8", "N11": "3.88", "N12": "3.96", "N13": "4.04", "N14": "4.12", "N15": "4.2", "N16": "4.28", "N17": "1.25", "N18": "1.3", "N19": "1.35", "N20": "1.4", "N21": "1.45", "N22": "1.5", "N23": "1.55", "N24": "1.6", "N25": "1.65", "N26": "1.7", "N27": "1.75", "N28": "1.8", "N29": "1.85", "N30": "1.9", "N31": "1.95"}
  ]
}