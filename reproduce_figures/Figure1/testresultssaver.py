import numpy as np
from results_saver import ResultsSaver
# Assuming ResultsSaver class is already defined or imported

def test_results_saver():
    # Create dummy results dictionary with numpy arrays
    results = {
        "array1": np.arange(10),
        "array2": np.random.rand(5, 3),
        "value": np.array(42)
    }

    # Simple parameters dictionary
    params = {
        "param1": 123,
        "param2": 3.14,
        "param3": "test_value",
        "param4": [1, 2, 3]
    }

    # Instantiate the saver and save the data
    saver = ResultsSaver()
    saver.save(results, params)

    print("Test complete. Check the results folder for saved files.")

if __name__ == "__main__":
    test_results_saver()
