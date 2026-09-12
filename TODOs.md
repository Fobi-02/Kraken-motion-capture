# TODOs

## Data manipulation

- [x] Function to plot the points with animations
- [ ] Use chassis points to rototranslate the chsssis in the correct RF (using model points) - Zano
- [x] Add wheel reference frame
- [ ] Correct missing kinematic points
- [x] Maps to assign two mocap points to a single suspension link with the correct name
        - Use plot_markers to associate the names at frame 0
        - Check if they are all present from the name of the columns of the csv file
        - Probably some markers appear in later frames so the json file also have to be updated
        - Pay attention to the steering markers (+ clockwise rotation, - counterclockwise rotation)
- [x] Function to transform df based on maps
- [x] Function to plot the chassis with links
- [x] Function to read the steering angle
- [x] Transform the kinematic model from wolfram to python

## Data analysis

- [ ] Compare kinematic maps
- [ ] Compare GPS positions
- [ ] Compare kinematic point positions
 
## To fix
- [x] Add support for other bumpsteer options for rear suspensions kinematics
- [x] Add reference frame for p9 -> correct the position along y
- [ ] When transofrming the coordinates in the correct RF also the P9 RF has to be transformed