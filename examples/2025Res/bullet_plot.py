import matplotlib.pyplot as plt
import numpy as np 
# Data
collisions = [34,24,32,26]     # Top black layer
captures = [6,1,5,-1]
grabs = [5,0,-2,-2]
policies = ['FeudalEffort \nvs H (Att,Def)', 'FeudalEffort \nvs RLB','zdg8672 \nvs H (Att,Def)', 'zdg8672 \nvs RLB']
# Plotting
plt.rc('font', size=22) 
width = 0.25
x = np.arange(len(policies))
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Bottom layer: Collisions
# ax.bar(policies, collisions, color='#e4ecf5', zorder=1, width=0.5, label='Collisions')

# # 2. Middle layer: Grabs
# ax.bar(policies, grabs, color='#5a8cc2', zorder=2, width=0.35, label='Grabs')

# # 3. Top layer: Score (Black, smallest)
# ax.bar(policies, captures, color='#2b4e72', zorder=3, width=0.2, label='Captures')
# ax.bar(policies, collisions, color='#e4ecf5', zorder=1, width=0.3, label='Collisions')

# # 2. Middle layer: Grabs
# ax.bar(policies, grabs, color='#5a8cc2', zorder=2, width=0.3, label='Grabs')

# # 3. Top layer: Score (Black, smallest)
# ax.bar(policies, captures, color='#2b4e72', zorder=3, width=0.3, label='Captures')
rects1 = ax.bar(x-width, collisions,width,label='Collisions',color='#e4ecf5')
rects2 = ax.bar(x, grabs,width,label='∆Grabs',color='#5a8cc2')
rects3 = ax.bar(x+width, captures,width,label='∆Captures',color='#2b4e72')
# Formatting
ax.set_xticks(x)
ax.set_xticklabels(policies)
ax.set_ylim(-5, 40)
ax.set_title("Algorithm Performance vs Baselines")
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
