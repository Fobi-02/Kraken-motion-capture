# TODOs

## Data manipulation

- [x] Function to plot the points with animations
- [x] Correctly connect points with colored links
- [ ] Use chassis points to rototranslate the chsssis in the correct RF (using model points). we use p1, p2, p3, p4 gps steering f and r, knowing they position w.r.t. the correct RF and for each points we derive a rototranslation matrix to get the points there with a minimization
- [x] Add wheel reference frame
- [x] Maps to assign two mocap points to a single suspension link with the correct name
        - Use plot_markers to associate the names at frame 0
        - Check if they are all present from the name of the columns of the csv file
        - Probably some markers appear in later frames so the json file also have to be updated
        - Pay attention to the steering markers (+ clockwise rotation, - counterclockwise rotation)
- [x] Function to transform df based on maps
- [x] Function to plot the chassis with links
- [x] Function to read the steering angle
- [x] Transform the kinematic model from wolfram to python
- [ ] ZANO: simplify marker_to_kinematic_point and do transofrm_RF 
- [ ] FEDE: modify steering angle and compare P9 RF with vehicle model

## Data analysis

- [ ] Find GPS positions w.r.t. the car RF
- [ ] Compare kinematic point positions on the chassis, mean of all the measurments over time to lewer uncertainty
- [ ] Compare kinematic maps of P9 RF, depending on delta and wheel height
 
## To fix
- [x] Add support for other bumpsteer options for rear suspensions kinematics
- [x] Add reference frame for p9 -> correct the position along y
- [ ] When transofrming the coordinates in the correct RF also the P9 RF has to be transformed
- [ ] when understanding a joint has only one marker instead of two, we use the position of that marker to get the position of the joint. Clearly this is wrong and we should just remove it
- [ ] ZANO: map_points.py is full of stuff and needs to be refactored. maybe add an utils.py as done for the plotting