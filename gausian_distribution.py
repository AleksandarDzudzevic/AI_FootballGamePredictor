import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the Gaussian distribution
mean_goals_per_match = 1.2  # Average goals per match
std_deviation = 0.5  # Standard deviation

# Generate x values (goals scored) for the plot
x = np.linspace(0, 3, 1000)  # Assuming a maximum of 3 goals per match

# Calculate the corresponding y values from the Gaussian distribution
y = norm.pdf(x, mean_goals_per_match, std_deviation)

# Plot the Gaussian distribution
plt.plot(x, y, label='Gaussian Distribution')
plt.xlabel('Goals Scored per Match')
plt.ylabel('Probability Density')
plt.title('Team\'s Goals Scored Distribution per Match')
plt.legend()
plt.grid()
plt.show()
