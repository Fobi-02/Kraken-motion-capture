from functions.read_csv import read_csv
from functions.plot_markers import plot_markers
from functions.plot_markers_slide import plot_markers_slider

csv_file = "../Data/F-30_001.csv"
df = read_csv(csv_file)

#plot_markers(df, 100)
plot_markers_slider(df, step=100)