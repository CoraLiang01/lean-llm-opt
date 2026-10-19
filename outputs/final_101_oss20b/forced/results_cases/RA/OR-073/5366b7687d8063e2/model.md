##### Objective Function: 

$\quad \quad \max \big( (1.25-0.25) \cdot (x_{I,A1} + x_{I,A2} + x_{I,A3} + x_{I,B1} + x_{I,B2} + x_{I,B3} + x_{I,B4}) + (2-0.35) \cdot (x_{II,A1} + x_{II,A2} + x_{II,A3} + x_{II,B1}) + (2.8-0.5) \cdot (x_{III,A2} + x_{III,B2}) \big) - \big( 300 \cdot (5 x_{I,A1} + 10 x_{II,A1}) + 321 \cdot (7 x_{I,A2} + 9 x_{II,A2} + 12 x_{III,A2}) + 203 \cdot (6 x_{I,A3} + 11 x_{II,A3}) + 250 \cdot (6 x_{I,B1} + 8 x_{II,B1}) + 783 \cdot (4 x_{I,B2} + 11 x_{III,B2}) + 200 \cdot (7 x_{I,B3}) + 300 \cdot (3 x_{I,B4} + 5 x_{II,B4}) \big)$

##### Constraints

###### 1. Equipment Time Constraints: 

$\quad \quad 5 x_{I,A1} + 10 x_{II,A1} \leq 6000$

$\quad \quad 7 x_{I,A2} + 9 x_{II,A2} + 12 x_{III,A2} \leq 10000$

$\quad \quad 6 x_{I,A3} + 11 x_{II,A3} \leq 8000$

$\quad \quad 6 x_{I,B1} + 8 x_{II,B1} \leq 4000$

$\quad \quad 4 x_{I,B2} + 11 x_{III,B2} \leq 7000$

$\quad \quad 7 x_{I,B3} \leq 4000$

$\quad \quad 3 x_{I,B4} + 5 x_{II,B4} \leq 5000$

###### 2. Non-negativity and Continuity Constraints: 

$\quad \quad  x_{I,A1}, x_{I,A2}, x_{I,A3}, x_{I,B1}, x_{I,B2}, x_{I,B3}, x_{I,B4}, x_{II,A1}, x_{II,A2}, x_{II,A3}, x_{II,B1}, x_{III,A2}, x_{III,B2} \in \mathbb{R}_{\geq 0}$

###### Retrieved Information

[
  {
    "Equipment / Cost": "A1",
    "Product I": "5",
    "Product II": "10",
    "Product III": "",
    "Available Equipment Operating Time": "6000",
    "Equipment Cost at Full Load (yuan)": "300"
  },
  {
    "Equipment / Cost": "A2",
    "Product I": "7",
    "Product II": "9",
    "Product III": "12",
    "Available Equipment Operating Time": "10000",
    "Equipment Cost at Full Load (yuan)": "321"
  },
  {
    "Equipment / Cost": "A3",
    "Product I": "6",
    "Product II": "11",
    "Product III": "2",
    "Available Equipment Operating Time": "8000",
    "Equipment Cost at Full Load (yuan)": "203"
  },
  {
    "Equipment / Cost": "B1",
    "Product I": "6",
    "Product II": "8",
    "Product III": "",
    "Available Equipment Operating Time": "4000",
    "Equipment Cost at Full Load (yuan)": "250"
  },
  {
    "Equipment / Cost": "B2",
    "Product I": "4",
    "Product II": "",
    "Product III": "11",
    "Available Equipment Operating Time": "7000",
    "Equipment Cost at Full Load (yuan)": "783"
  },
  {
    "Equipment / Cost": "B3",
    "Product I": "7",
    "Product II": "",
    "Product III": "",
    "Available Equipment Operating Time": "4000",
    "Equipment Cost at Full Load (yuan)": "200"
  },
  {
    "Equipment / Cost": "B4",
    "Product I": "3",
    "Product II": "5",
    "Product III": "8",
    "Available Equipment Operating Time": "5000",
    "Equipment Cost at Full Load (yuan)": "300"
  },
  {
    "Equipment / Cost": "Raw Material Cost (yuan/unit)",
    "Product I": "0.25",
    "Product II": "0.35",
    "Product III": "0.5",
    "Available Equipment Operating Time": "",
    "Equipment Cost at Full Load (yuan)": ""
  },
  {
    "Equipment / Cost": "Unit Price (yuan/unit)",
    "Product I": "1.25",
    "Product II": "2",
    "Product III": "2.8",
    "Available Equipment Operating Time": "",
    "Equipment Cost at Full Load (yuan)": ""
  }
]