
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text

# read csv dataset for grawing graphs
dataset = pd.read_csv('E:/forth year project/final attribute table/final attribute table.csv')

print(dataset.columns)

# class TextHandler(HandlerBase):
#     def create_artists(self, legend, tup ,xdescent, ydescent,
#                         width, height, fontsize,trans):
#         tx = Text(width/2.,height/2,tup[0], fontsize=fontsize,
#                   ha="center", va="center", color=tup[1], fontweight="bold")
#         return [tx]
#
#
# a = np.random.choice(["1", "2", "3", "4", "5", "6","7", "8", "9", "10"], size=100,
#                      p=np.arange(1,11)/21. )
# df = pd.DataFrame(a, columns=["Distance_to_Stream"])
# ax = sns.countplot(x = df.Distance_to_Stream)
# plt.title('Distances To Streams')
# handltext = ["1", "2", "3", "4", "5", "6","7", "8", "9", "10"]
# labels = ["0-100 m", "100-200 m", "200-300 m", "300-400 m", "400-500 m", "500-600 m", "600-700 m", "700-800 m", "800-900 m", "900<= m"]
#
# t = ax.get_xticklabels()
# labeldic = dict(zip(handltext, labels))
# labels = [labeldic[h.get_text()]  for h in t]
# handles = [(h.get_text(),c.get_fc()) for h,c in zip(t,ax.patches)]
# ax.legend(handles, labels, handler_map={tuple : TextHandler()})


sns.countplot(x='Distance_to_Stream',data=dataset,palette='hls')
plt.xticks(range(10),["0-100 m", "100-200 m", "200-300 m", "300-400 m", "400-500 m", "500-600 m", "600-700 m", "700-800 m", "800-900 m", "900<= m"] ,rotation='vertical')
plt.title('Distances To Streams')
plt.show()


sns.countplot(x='Distance_to_Road',data=dataset,palette='hls')
plt.xticks(range(10),["0-100 m", "100-200 m", "200-300 m", "300-400 m", "400-500 m", "500-600 m", "600-700 m", "700-800 m", "800-900 m", "900<= m"] ,rotation='vertical')
plt.title('Distances To Roads')
plt.show()

sns.countplot(x='Plane_Curvature',data=dataset,palette='hls')
# plt.xticks(range(10),["0-100 m", "100-200 m", "200-300 m", "300-400 m", "400-500 m", "500-600 m", "600-700 m", "700-800 m", "800-900 m", "900<= m"] ,rotation='vertical')
plt.title('Changes Of Plane Curvature')
plt.show()

sns.countplot(x='Profile_Curvature',data=dataset,palette='hls')
plt.title('Changes Of Profile Curvature')
plt.show()

sns.countplot(x='Slope_Angle',data=dataset,palette='hls')
plt.xticks(range(10),["0-4.059 ", "4.059-10.486", "10.486-16.575", "16.575-22.325", "22.325-28.086", "28.086-34.165", "34.165-41.945", "41.945-54.460", "54.460-61.068", "61.068<= "] ,rotation='vertical')
plt.title('Changes Of Slope Angle')
plt.show()

sns.countplot(x='Slope_Aspect',data=dataset,palette='hls')
plt.xticks(range(10),["F", "N", "NE", "E", "SE", "S", "SW", "W", "NW"])
plt.title('Changes Of Slope Aspect')
plt.show()










# plt.xticks(range(10),["fdf","fdsf","sdsa","ewewq","weqw","rter","sdsa","ewewq","weqw","rter"] ,rotation='vertical')