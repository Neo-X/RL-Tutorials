import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure()
plt.axis([-10, 10, -10, 10])

# Define properties of the "bouncing balls"
n = 10
pos = (20 * np.random.sample(n*2) - 10).reshape(n, 2)
vel = (0.3 * np.random.normal(size=n*2)).reshape(n, 2)
sizes = 100 * np.random.sample(n) + 100

# Colors where each row is (Red, Green, Blue, Alpha).  Each can go
# from 0 to 1.  Alpha is the transparency.
colors = np.random.sample([n, 4])

# Draw all the circles and return an object ``circles`` that allows
# manipulation of the plotted circles.
circles = plt.scatter(pos[:,0], pos[:,1], s=sizes, marker='o', c=colors)

elastic = 0.95
gravity = 0.05
length=1000
for i in range(length):
	pos = pos + vel
	bounce = abs(pos) > 10      # Find balls that are outside walls
	vel[bounce] = -elastic * vel[bounce]  # Bounce if outside the walls
	vel[:,1] -= gravity	
	circles.set_offsets(pos) 
    # The velocity vector for each ball.
	# for j in range(n):	
		# vector = arrow(pos[j,0], pos[j,1], vel[j,0], vel[j,1], color=colors[j])
	plt.draw()
