import pandas as pd
import matplotlib.pyplot as plt

# Set the path to your CSV file
loss_path = 'csv_files/loss_curve.csv'

# Use the read_csv() function from Pandas to read the file
data_loss = pd.read_csv(loss_path)

# Display the data
print(data_loss)


plt.plot(data_loss['Iteration'],data_loss['L1_loss'], '.')
plt.show()