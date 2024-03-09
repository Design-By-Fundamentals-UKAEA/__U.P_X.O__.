from timer import Timer

# METHOD - 1
t = Timer()
t.start()
points = []
for lat in np.arange(-1, 1, 0.001):
    for lon in np.arange(-1, 1, 0.001):
        points.append([lat, lon])
t.stop()
#Elapsed time: 2.23 to 2.89 seconds

# METHOD - 2
t = Timer()
t.start()
points = [[_x, _y] for _x in np.arange(-1, 1, 0.001) for _y in np.arange(-1, 1, 0.001)]
t.stop()
#Elapsed time: 1.7 to 2.20 seconds

# METHOD - 3
t = Timer()
t.start()
x = np.arange(-1, 1, 0.001)
y = np.arange(-1, 1, 0.001)
x, y = np.meshgrid(x, y)
t.stop()
# Elapsed time: 0.0190 seconds : Consistant