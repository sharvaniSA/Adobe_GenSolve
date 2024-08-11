import numpy as np

def read_csv(csv_path):
    # Load the data from the CSV file
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    
    # Initialize an empty list to hold the paths
    path_XYs = []
    
    # Extract unique paths
    for i in np.unique(np_path_XYs[:, 0]):
        # Filter points for the current path
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        # Initialize an empty list to hold the points for the current path
        XYs = []
        
        # Extract unique points
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        
        # Add the current path to the list of paths
        path_XYs.append(XYs)
    
    return path_XYs

# Example usage
csv_path = './problems/isolated.csv'
paths = read_csv(csv_path)
print(paths)