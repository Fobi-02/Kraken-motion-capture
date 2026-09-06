# TODOs

## Data manipulation

- [x] Function to plot the points with animations
- [ ] Function to transform coordinates from mocap RF to vehicle RF
- [ ] Minimization to associate all static (chassis) kinematic points
- [ ] Function to compensate for missing data of certain points
- [ ] Maps to assign two mocap points to a single suspension link with the correct name (2/17)
        - Use plot_markers to associate the names at frame 0
        - Check if they are all present from the name of the columns of the csv file
        - Probably some markers appear in later frames so the json file also have to be updated
        - Pay attention to the steering markers (+ clockwise rotation, - counterclockwise rotation)
- [x] Function to transform df based on maps
- [x] Function to plot the chassis with links
- [ ] Function to read the steering angle
- [x] Transform the kinematic model from wolfram to python

## Data analysis

- [ ] Compare kinematic maps

## To fix
- [ ] Add support for other bumpsteer options for rear suspensions kinematics
- [x] Add reference frame for p9 -> correct the position along y