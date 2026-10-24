{
  "query": "A paper mill cuts standard rolls into smaller item types. The demand for each item type is provided in item_demand.csv. A set of feasible cutting patterns is provided in cutting_patterns.csv, where each pattern specifies how many units of each item type are produced by cutting one standard roll according to that pattern.\n\nFormulate an integer cutting-stock pattern-selection model. For each cutting pattern p, define y_p as the integer number of standard rolls cut using pattern p. The objective is to minimize the total number of standard rolls used. The model should satisfy or exceed demand for every item type, impose nonnegativity on pattern-use variables, and require pattern-use variables to be integers.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Item",
        "Demand"
      ],
      "file_name": "item_demand.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Demand": "24",
            "Item": "A"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Demand": "18",
            "Item": "B"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Demand": "12",
            "Item": "C"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Demand": "10",
            "Item": "D"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Pattern",
        "A",
        "B",
        "C",
        "D"
      ],
      "file_name": "cutting_patterns.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "A": "4",
            "B": "0",
            "C": "0",
            "D": "0",
            "Pattern": "P1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "A": "0",
            "B": "3",
            "C": "0",
            "D": "0",
            "Pattern": "P2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "A": "0",
            "B": "0",
            "C": "2",
            "D": "0",
            "Pattern": "P3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "A": "0",
            "B": "0",
            "C": "0",
            "D": "2",
            "Pattern": "P4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "A": "2",
            "B": "1",
            "C": "0",
            "D": "0",
            "Pattern": "P5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "A": "1",
            "B": "0",
            "C": "1",
            "D": "0",
            "Pattern": "P6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "A": "0",
            "B": "1",
            "C": "0",
            "D": "1",
            "Pattern": "P7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "A": "1",
            "B": "1",
            "C": "1",
            "D": "0",
            "Pattern": "P8"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "A": "2",
            "B": "0",
            "C": "0",
            "D": "1",
            "Pattern": "P9"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
