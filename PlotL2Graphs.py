import pandas as pd
import matplotlib.pyplot as plt


#### THIS FILE IS OUR OF DATE - WE ARE USING LOCAL VERSION ####

# Set the path to your CSV file
loss_path = 'csv_files/loss_curve.csv'

# Use the read_csv() function from Pandas to read the file
data_loss = pd.read_csv(loss_path)

# Display the data
print(data_loss)


plt.figure()
plt.plot(data_loss['Iteration'],data_loss['L1_loss'], '.')
plt.title('L1_Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('L1_loss')
plt.grid()
plt.show()

plt.figure()
plt.plot(data_loss['Iteration'],data_loss['MS_SSIM'], '.')
plt.title('MS_SSIM Curve')
plt.xlabel('Iteration')
plt.ylabel('MS_SSIM')
plt.grid()
plt.show()