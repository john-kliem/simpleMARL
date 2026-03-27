import matplotlib.pyplot as plt
import numpy as np 
# Data
captures = [-10, 12, -2, -2]
grabs = [-17, 15, -11, -2]
policies = ['Controller\n(Heuristic + DRL)', 'Subsumption +\n Controller(Heuristic)', 'Subsumption +\n Controller(DRL)', 'Subsumption']
# Plotting
plt.rc('font', size=22) 
width = 0.25
x = np.arange(len(policies))
fig, ax = plt.subplots(figsize=(10, 6))


rects2 = ax.bar(x-width, grabs,width,label='∆Grabs',color='#5a8cc2')
rects3 = ax.bar(x, captures,width,label='∆Captures',color='#2b4e72')
# Formatting
ax.set_xticks(x)
ax.set_xticklabels(policies)
ax.set_ylim(-20, 20)
ax.set_title("FeudalEffort Ablation")
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
