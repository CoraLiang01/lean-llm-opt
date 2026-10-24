{
  "query": "There are multiple suppliers responsible for delivering essential goods daily to customer groups located in different regions. Each supplier has a specific daily supply capacity, detailed in \"supply_capacity.csv\", while the daily demand of each customer group is recorded in \"customer_demand.csv\". The transportation cost per unit of goods from each supplier to each customer group is provided in \"transportation_costs.csv\". The objective is to determine the optimal transportation plan, specifying how goods should be allocated from each supplier to each customer group, ensuring that all demands are met without exceeding the supply capacity of any supplier, while minimizing the total transportation cost.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "customer",
        "demand"
      ],
      "file_name": "customer_demand.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "customer": "C1",
            "demand": "216"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "customer": "C2",
            "demand": "168"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "customer": "C3",
            "demand": "264"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "customer": "C4",
            "demand": "216"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "customer": "C5",
            "demand": "216"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "customer": "C6",
            "demand": "192"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "customer": "C7",
            "demand": "144"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "customer": "C8",
            "demand": "168"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "customer": "C9",
            "demand": "168"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "customer": "C10",
            "demand": "168"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Unnamed: 0",
        "supply_capacity"
      ],
      "file_name": "supply_capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Unnamed: 0": "S1",
            "supply_capacity": "288"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Unnamed: 0": "S2",
            "supply_capacity": "288"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Unnamed: 0": "S3",
            "supply_capacity": "264"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Unnamed: 0": "S4",
            "supply_capacity": "264"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Unnamed: 0": "S5",
            "supply_capacity": "216"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Unnamed: 0": "S6",
            "supply_capacity": "216"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Unnamed: 0": "S7",
            "supply_capacity": "168"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Unnamed: 0": "S8",
            "supply_capacity": "216"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Unnamed: 0": "S9",
            "supply_capacity": "240"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Unnamed: 0": "S10",
            "supply_capacity": "168"
          }
        }
      ],
      "table_id": "file_1_view_0"
    },
    {
      "columns": [
        "Unnamed: 0",
        "C1",
        "C2",
        "C3",
        "C4"
      ],
      "file_name": "transportation_costs.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "C1": "590.3648136504455",
            "C2": "23.669172607322494",
            "C3": "88.89005869765714",
            "C4": "497.52228807074613",
            "Unnamed: 0": "S1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "C1": "2042.0715001593626",
            "C2": "2133.978484314172",
            "C3": "705.15912033561",
            "C4": "101.59454516295598",
            "Unnamed: 0": "S2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "C1": "22.297222160217984",
            "C2": "497.9271939314995",
            "C3": "1653.0828862073263",
            "C4": "23.68545123339267",
            "Unnamed: 0": "S3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "C1": "960.7814533858373",
            "C2": "49.128300053752405",
            "C3": "1324.238697073691",
            "C4": "1032.2095478151716",
            "Unnamed: 0": "S4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "C1": "1471.2721666392908",
            "C2": "85.6956072820555",
            "C3": "38.89266823851542",
            "C4": "1542.0500358120464",
            "Unnamed: 0": "S5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "C1": "191.9058726130392",
            "C2": "158.50401031820448",
            "C3": "91.02045349777458",
            "C4": "184.44747201726193",
            "Unnamed: 0": "S6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "C1": "81.23891457326876",
            "C2": "0.3744642223062507",
            "C3": "2079.46686537067",
            "C4": "0.3065671755503025",
            "Unnamed: 0": "S7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "C1": "56.099310965461356",
            "C2": "935.6143108671334",
            "C3": "73.08824617002863",
            "C4": "52.00392409272077",
            "Unnamed: 0": "S8"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "C1": "4.502283326860296",
            "C2": "0.3899585342810754",
            "C3": "1782.4662178163346",
            "C4": "0.006345906612718274",
            "Unnamed: 0": "S9"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "C1": "333.6869270439132",
            "C2": "277.4719386113677",
            "C3": "86.02096892455509",
            "C4": "277.30836609256806",
            "Unnamed: 0": "S10"
          }
        }
      ],
      "table_id": "file_2_view_0"
    }
  ]
}
