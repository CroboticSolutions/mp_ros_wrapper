import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
  filename = 'desni_kaziprst.csv'

  data = pd.read_csv(filename)

  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')


  xs = data['r_index_x']
  ys = data['r_index_y']
  zs = data['r_index_z']

  ax.plot(xs, ys, zs, label='Putanja prsta', color='blue', linewidth=2)
  ax.scatter(xs, ys, zs, c='red', s=20)
  for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
      label = str(i)  
      ax.text(x, y, z, label, fontsize=8) 

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('3D trajectory')
  ax.legend()

  plt.show()

if __name__ == "__main__":
    main()