import os
import numpy as np

# Directory where the text files are stored
directory = r'C:\Development\M2MatMod\UPXO-CPFEM-Preperation\mcgs 100x100x100 S16\texture instance 1 mctime step 49\Props'

delimiter = ','  # Change this to match the delimiter used in your .dat files
fileExtension = '.txt'

data_UPXO_MCGS3D = {}

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a .dat file
    if filename.endswith(fileExtension):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Read the numerical values from the file and store them in a NumPy array
        with open(file_path, 'r') as file:
            # Load the data from the file using the specified delimiter
            content_array = np.loadtxt(file, delimiter=delimiter)
            # Extract the filename without the extension and use it as the key
            key = os.path.splitext(filename)[0]
            # Add to the dictionary
            data_UPXO_MCGS3D[key] = content_array

# data_UPXO_MCGS3D now holds the content of each .dat file as a NumPy array, keyed by the filename without extension
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# Assuming data_UPXO_MCGS3D['EquivalentDiameters'] contains a 1D NumPy array of numerical values
data = data_UPXO_MCGS3D['EquivalentDiameters']

# Step 1: Construct the KDE
kde = gaussian_kde(data)
kde.set_bandwidth(bw_method=kde.factor / 2)
x_grid = np.linspace(data.min(), data.max(), 1000)
kde_values = kde.evaluate(x_grid)

# Step 2: Identify the peaks in the KDE
peaks, _ = find_peaks(kde_values)

# Step 3: Plot the KDE with identified peaks
plt.figure(figsize=(8, 6))
plt.plot(x_grid, kde_values, label='KDE')
plt.scatter(x_grid[peaks], kde_values[peaks], color='red', marker='o', s=50, label='Peaks')
plt.title('KDE with Identified Peaks')
plt.xlabel('Equivalent Diameters')
plt.ylabel('Density')
plt.legend()
plt.show()




import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Assuming data_UPXO_MCGS3D['EquivalentDiameters'] contains a 1D NumPy array of numerical values
data = data_UPXO_MCGS3D['EquivalentDiameters']

# Step 1: Plot the KDE using Seaborn
plt.figure(figsize=(8, 6))
sns_kde = sns.kdeplot(data, bw_adjust=0.5, kernel = 'gau',cumulative=False, linewidth=1, color='blue',label='KDE')

# Step 2: Extract the lines from the plot to find peaks
line = sns_kde.lines[0]
x, y = line.get_data()

# Step 3: Identify the peaks in the KDE
peaks, _ = find_peaks(y)

# Step 4: Plot the identified peaks on the KDE plot
plt.scatter(x[peaks], y[peaks], color='red', marker='o', s=50, label='Peaks')
plt.title('KDE with Identified Peaks')
plt.xlabel('Equivalent Diameters')
plt.ylabel('Density')
plt.legend()
plt.show()
