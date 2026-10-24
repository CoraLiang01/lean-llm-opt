{
  "query": "A boat dealership needs to allocate different types of boats into different display areas. Specifically, the dealership has several display areas, each with a capacity limit provided in “capacity.csv.” The predefined value and size of each boat type can be found in “products.csv.” The objective is to determine the optimal number of units of each boat type to place in each display area to maximize the total value of the boats across all areas, while ensuring that the total size of the boats in each area does not exceed its capacity. The decision variables x_{ij} represent the number of units of boat type j to be placed in display area i.The decision variables must be integers.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "DisplayID",
        "Capacity"
      ],
      "file_name": "capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Capacity": "356",
            "DisplayID": "1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Capacity": "478",
            "DisplayID": "2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Capacity": "305",
            "DisplayID": "3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Capacity": "291",
            "DisplayID": "4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Capacity": "168",
            "DisplayID": "5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Capacity": "449",
            "DisplayID": "6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Capacity": "139",
            "DisplayID": "7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Capacity": "383",
            "DisplayID": "8"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Capacity": "472",
            "DisplayID": "9"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Capacity": "288",
            "DisplayID": "10"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "Capacity": "320",
            "DisplayID": "11"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "Capacity": "250",
            "DisplayID": "12"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "Capacity": "402",
            "DisplayID": "13"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "Capacity": "293",
            "DisplayID": "14"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "ProductName",
        "Value",
        "Weight"
      ],
      "file_name": "products.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "ProductName": "Speedboat",
            "Value": "69978",
            "Weight": "18"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "ProductName": "Fishing Boat",
            "Value": "54011",
            "Weight": "42"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "ProductName": "Catamaran",
            "Value": "36352",
            "Weight": "49"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "ProductName": "Yacht",
            "Value": "51521",
            "Weight": "42"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "ProductName": "Sailboat",
            "Value": "50415",
            "Weight": "41"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "ProductName": "Kayak",
            "Value": "76109",
            "Weight": "48"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "ProductName": "Canoe",
            "Value": "50462",
            "Weight": "22"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "ProductName": "Houseboat",
            "Value": "28989",
            "Weight": "29"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "ProductName": "Pontoon",
            "Value": "23318",
            "Weight": "45"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "ProductName": "Jet Ski",
            "Value": "26142",
            "Weight": "14"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "ProductName": "Rowboat",
            "Value": "42040",
            "Weight": "38"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "ProductName": "Hovercraft",
            "Value": "85961",
            "Weight": "47"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "ProductName": "Cabin Cruiser",
            "Value": "50142",
            "Weight": "45"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "ProductName": "Wakeboard Boat",
            "Value": "48478",
            "Weight": "28"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "ProductName": "Dinghy",
            "Value": "60953",
            "Weight": "24"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "ProductName": "Trawler",
            "Value": "95265",
            "Weight": "39"
          }
        },
        {
          "data_row_number": 16,
          "values": {
            "ProductName": "Paddle Boat",
            "Value": "22839",
            "Weight": "32"
          }
        },
        {
          "data_row_number": 17,
          "values": {
            "ProductName": "Submarine",
            "Value": "90957",
            "Weight": "36"
          }
        },
        {
          "data_row_number": 18,
          "values": {
            "ProductName": "RIB",
            "Value": "84652",
            "Weight": "14"
          }
        },
        {
          "data_row_number": 19,
          "values": {
            "ProductName": "Skiff",
            "Value": "78991",
            "Weight": "16"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
