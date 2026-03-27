import matplotlib.pyplot as plt

# Data for the first pie chart (2 categories)
labels1 = ['Subsumption', 'Controller']
sizes1 = [37.7, 62.2]
#explode1 = (0, 0.1)  # only "explode" the 2nd slice

# Data for the second pie chart (5 categories)
labels2 = ['Heuristic', 'DRL']
sizes2 = [73.2, 26.7]
#explode2 = (0, 0.1, 0, 0, 0)  # only "explode" the 2nd slice

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.rc('font', size=22) 
# Plot the first pie chart
ax1.pie(sizes1, labels=labels1, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=['#5a8cc2','#b2cbde'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Percentage of\nAction Selection')

# Plot the second pie chart
ax2.pie(sizes2, labels=labels2, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=['#d8e1e8', '#98bad5'])
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax2.set_title('Percentage of Action \nSelection within Controller')

# Display the charts
plt.tight_layout()
plt.show()
