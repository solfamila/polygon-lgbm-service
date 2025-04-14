try:
    import pandas_ta as ta
    print("Successfully imported pandas_ta!")
    # Optional: Try using a function
    # import pandas as pd
    # df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
    # df.ta.rsi(append=True)
    # print("RSI calculation successful.")
    # print(df)
except ImportError as e:
    print(f"Failed to import pandas_ta: {e}")
    import sys
    print("\nsys.path:")
    import pprint
    pprint.pprint(sys.path)
except Exception as e_other:
     print(f"An unexpected error occurred: {e_other}")
