import matplotlib.pyplot as plt
gs = spvo.PXTALS_list[0][0][0]

plt.figure(figsize = (3.5, 3.5), dpi = 200)
gcount = 0
for grain in gs.geoms:
    plt.fill(grain.boundary.xy[0],
             grain.boundary.xy[1],
             color = 'white',
             edgecolor = 'black',
             linewidth = 2)
    xc = grain.centroid.x
    yc = grain.centroid.y
    plt.text(xc, yc, str(gcount), fontsize = 5)
    gcount += 1