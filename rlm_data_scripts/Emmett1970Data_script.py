"""
Purpose: Plot series 8 lab data from Emmett 1970 in a nice way.
"""
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data and split into separate tests
data_df = pd.read_csv('./rlm_input/Emmett1970_data.csv', index_col='Test')
data_2_df = pd.read_csv('./rlm_input/NF2_data.csv')
test8_13 = data_df[data_df.index=='8-13']
test8_14 = data_df[data_df.index=='8-14']
test8_15 = data_df[data_df.index=='8-15']
test8_16 = data_df[data_df.index=='8-16']
test8_17 = data_df[data_df.index=='8-17']

# Add line of best fit (without doing fancy numpy math 
# since I already have the equation...)
q1 = np.arange(1e-6,1,1e-6)
n1 = 0.0026*q1**(-0.274)

q2 = np.arange(1e-6,0.001,1e-6)
n2 = 0.08*q2**(-0.153)

# Close previous plots
plt.close('all')

# Set up the figure and axis parameters
fig, ax = plt.subplots(figsize=(6.5,4.5))
ax.tick_params(bottom=True, top=True, left=True, right=True, which='both')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((1e-6,1e-3))
ax.set_ylim((0.01,0.10))
ax.grid(True, which='both', color='gainsboro')
ax.set_axisbelow(True)

# Plot the data
ax.plot(q1, n1, ':', c='indigo',label='Line of best fit')
ax.scatter(test8_13.discharge_cms, test8_13.n_SI, label='8-13', marker='s', 
	edgecolor='darkgreen', c='limegreen')
ax.scatter(test8_14.discharge_cms, test8_14.n_SI, label='8-14', marker='D', 
	edgecolor='royalblue', c='lightskyblue')
ax.scatter(test8_15.discharge_cms, test8_15.n_SI, label='8-15', marker='^', 
	edgecolor='darkgoldenrod', c='gold')
ax.scatter(test8_16.discharge_cms, test8_16.n_SI, label='8-16', marker='o', 
	edgecolor='orangered', c='darkorange')
ax.scatter(test8_17.discharge_cms, test8_17.n_SI, label='8-17', marker='*', 
	edgecolor='mediumvioletred', c='hotpink', 
	s=49)

# Set labels, legend, text, title
ax.set_xlabel(r'Discharge per unit width, $q$ [$\frac{m^2}{s}$]', fontsize=12)
ax.set_ylabel('Manning\'s n', fontsize=12)
ax.text(x=1.15e-6, y=6.25e-2, s=r'$n = 0.0026q^{-0.274}$')
ax.legend(loc='lower left')
ax.set_title('Series 8 Laboratory Data', fontsize=14)

# Show plot and save
plt.tight_layout()
plt.show()
# plt.savefig(r'./rlm_output/S8_data.png')

# ----------------------------------------------------------------------------
# Set up the figure and axis parameters
fig1, ax1 = plt.subplots(figsize=(6.5,4.5))
ax1.tick_params(bottom=True, top=True, left=True, right=True, which='both')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim((1e-6,0.001))
ax1.set_ylim((0.01,10.00))
ax1.grid(True, which='both', color='gainsboro')
ax1.set_axisbelow(True)

# Plot the data
ax1.plot(q2, n2, ':', c='maroon',label='Line of best fit')
ax1.plot(data_2_df.q, data_2_df.n, '-o', color= 'royalblue',
	markeredgecolor='midnightblue', 
	markerfacecolor='royalblue')

# Set labels, legend, text, title
ax1.set_xlabel(r'Discharge per unit width, $q$ [$\frac{m^2}{s}$]', fontsize=12)
ax1.set_ylabel('Manning\'s n', fontsize=12)
ax1.text(x=2.0e-6, y=0.80, s=r'$n = 0.08q^{-0.153}$')
ax1.set_title('New Fork River Site 2 Data', fontsize=14)

# Show plot and save
plt.tight_layout()
plt.show()
# plt.savefig(r'./rlm_output/NF2_data.png')