'''
Conversion from wolfram mathematica script to python
Script used to study suspension kinematics
'''

import json
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

#  _____       _        _   _               __  __       _        _      
# |  __ \     | |      | | (_)             |  \/  |     | |      (_)     
# | |__) |___ | |_ __ _| |_ _  ___  _ __   | \  / | __ _| |_ _ __ ___  __
# |  _  // _ \| __/ _` | __| |/ _ \| '_ \  | |\/| |/ _` | __| '__| \ \/ /
# | | \ \ (_) | || (_| | |_| | (_) | | | | | |  | | (_| | |_| |  | |>  < 
# |_|  \_\___/ \__\__,_|\__|_|\___/|_| |_| |_|  |_|\__,_|\__|_|  |_/_/\_\
                                                                        
def translate(x, y, z):
    """Return a 4x4 translation matrix."""
    return np.array([
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ])

def rotate(axis, theta):
    """Return a 4x4 rotation matrix."""
    c = sp.cos(theta)
    s = sp.sin(theta)
    axis = axis.upper()

    if axis == "X":
        return np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, c, -s, 0.0],
             [0.0, s, c, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )

    if axis == "Y":
        return np.array(
            [[c, 0.0, s, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [-s, 0.0, c, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )

    if axis == "Z":
        return np.array(
            [[c, -s, 0.0, 0.0],
             [s, c, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )

    raise ValueError("axis must be 'X', 'Y', or 'Z'")

def get_point(rf):
    """Extract the origin of a reference frame."""
    return sp.Matrix(rf[:3, 3].copy())

def make_point(rf, x, y, z):
    """Create a point expressed in RF and return it in ground coordinates."""
    return sp.Matrix(get_point(np.asarray(rf) @ translate(x, y, z)))

def inv_frame(rf):
    """Return the inverse of a rigid homogeneous reference frame."""
    rf = np.asarray(rf, dtype=float)
    rm = rf[:3, :3].T
    po = -rm @ rf[:3, 3]

    result = np.eye(4)
    result[:3, :3] = rm
    result[:3, 3] = po
    return result

def intersection(p1, p2, p3, p4):
    p1x, p1y = p1
    p2x, p2y = p2
    p3x, p3y = p3
    p4x, p4y = p4

    denominator = (p1y * p3x - p2y * p3x - p1x * p3y + p2x * p3y
        - p1y * p4x + p2y * p4x + p1x * p4y - p2x * p4y)

    x = -(-p1y * p2x * p3x + p1x * p2y * p3x
        + p1y * p2x * p4x - p1x * p2y * p4x
        + p1x * p3y * p4x - p2x * p3y * p4x
        - p1x * p3x * p4y + p2x * p3x * p4y) / denominator

    y = -(p1y * p2x * p3y - p1x * p2y * p3y
        - p1y * p3y * p4x + p2y * p3y * p4x
        - p1y * p2x * p4y + p1x * p2y * p4y
        + p1y * p3x * p4y - p2y * p3x * p4y) / (-denominator)

    return np.array([x, y])

class RearKinematics:

    def __init__(self):
        # setup
        self.gamma0 = 0
        self.delta0 = 0
        # symbolic variables (wheel and chassis coordinates)
        self.xWl, self.yWl, self.zWl, self.deltaWl, self.gammaWl, self.thetaWl = sp.symbols('xWl yWl zWl deltaWl gammaWl thetaWl')
        self.xWr, self.yWr, self.zWr, self.deltaWr, self.gammaWr, self.thetaWr = sp.symbols('xWr yWr zWr deltaWr gammaWr thetaWr')
        self.DeltaZ, self.phi = sp.symbols('DeltaZ phi')

    def set_kinematic_problem(self):
        '''
        Writes the kinematic problem and returns the system of equations
        '''
        #  _____        _        
        # |  __ \      | |       
        # | |  | | __ _| |_ __ _ 
        # | |  | |/ _` | __/ _` |
        # | |__| | (_| | || (_| |
        # |_____/ \__,_|\__\__,_|

        #region Data
        
        # importing vehicle data and suspension points
        with open("VehicleData.json", "r", encoding="utf-8") as f:
            VehicleData = json.load(f)
        globals().update(VehicleData)

        with open("RearSuspensionPoints.json", "r", encoding="utf-8") as f:
                dataKine = json.load(f)
        globals().update(dataKine)

        # setup
        gamma0 = self.gamma0
        delta0 = self.delta0
        deltaZ0 = 0

        #endregion

        #  _  ___                            _   _                          _ _               
        # | |/ (_)                          | | (_)                        | (_)              
        # | ' / _ _ __   ___ _ __ ___   __ _| |_ _  ___    __ _ _ __   __ _| |_ ___ _   _ ___ 
        # |  < | | '_ \ / _ \ '_ ` _ \ / _` | __| |/ __|  / _` | '_ \ / _` | | / __| | | / __|
        # | . \| | | | |  __/ | | | | | (_| | |_| | (__  | (_| | | | | (_| | | \__ \ |_| \__ \
        # |_|\_\_|_| |_|\___|_| |_| |_|\__,_|\__|_|\___|  \__,_|_| |_|\__,_|_|_|___/\__, |___/
        #                                                                            __/ |    
        #                                                                           |___/     

        #region Kinematic definitions

        # Ground, road surface
        RF0 = np.eye(4)

        # =============================================================================
        # Hub + wheel
        # =============================================================================

        # wheel reference frames
        RFhl = translate(x9 + self.xWl, y9 + self.yWl, z9 + self.zWl) @ rotate("Z", delta0 + self.deltaWl) @ rotate("X", -gamma0 + self.gammaWl) @ rotate("Y", self.thetaWl)
        RFhr = translate(x9 + self.xWr, -y9 + self.yWr, z9 + self.zWr) @ rotate("Z", delta0 + self.deltaWr) @ rotate("X", -gamma0 + self.gammaWr) @ rotate("Y", self.thetaWr)

        # Points defined in the hub bracket reference frame
        P9l = get_point(RFhl) # Wheel center
        PCl = make_point(RFhl, xC, yC, zC) # Contact point with ground
        P6l = make_point(RFhl, x6, y6, z6) # Upper outboard attatchment point
        P7l = make_point(RFhl, x7, y7, z7) #Lower outboard attatchment point
        P8l = make_point(RFhl, x8, y8, z8) #Outboard tie rod attatchment point

        P9r = get_point(RFhr)
        PCr = make_point(RFhr, xC, -yC, zC)
        P6r = make_point(RFhr, x6, -y6, z6)
        P7r = make_point(RFhr, x7, -y7, z7)
        P8r = make_point(RFhr, x8, -y8, z8)

        # =============================================================================
        # Chassis
        # =============================================================================

        # chassis reference frame
        RFc = RF0 @ translate(0, 0, deltaZ0) @ rotate("X", self.phi)

        # four inboard pickup points (defined in the chassis RF)
        P1l = make_point(RFc, x1, y1, z1)
        P2l = make_point(RFc, x2, y2, z2)
        P3l = make_point(RFc, x3, y3, z3)
        P4l = make_point(RFc, x4, y4, z4)
        P5l = make_point(RFc, x5, y5, z5)

        P1r = make_point(RFc, x1, -y1, z1)
        P2r = make_point(RFc, x2, -y2, z2)
        P3r = make_point(RFc, x3, -y3, z3)
        P4r = make_point(RFc, x4, -y4, z4)
        P5r = make_point(RFc, x5, -y5, z5)

        #endregion

        #region Initial configuration

        initConf = {
            self.DeltaZ : 0,
            self.phi : 0,
            self.xWl : 0,
            self.xWr : 0,
            self.yWl : 0,
            self.yWr : 0,
            self.zWl : 0,
            self.zWr : 0,
            self.gammaWl : 0,
            self.gammaWr : 0, 
            self.thetaWl : 0,
            self.thetaWr : 0, 
            self.deltaWl : 0, 
            self.deltaWr : 0
        }

        # points in the initial configuration
        P1l0 = P1l.subs(initConf)
        P2l0 = P2l.subs(initConf)
        P3l0 = P3l.subs(initConf)
        P4l0 = P4l.subs(initConf)
        P5l0 = P5l.subs(initConf)
        P6l0 = P6l.subs(initConf)
        P7l0 = P7l.subs(initConf)
        P8l0 = P8l.subs(initConf)
        P9l0 = P9l.subs(initConf)
        PCl0 = PCl.subs(initConf)

        P1r0 = P1r.subs(initConf)
        P2r0 = P2r.subs(initConf)
        P3r0 = P3r.subs(initConf)
        P4r0 = P4r.subs(initConf)
        P5r0 = P5r.subs(initConf)
        P6r0 = P6r.subs(initConf)
        P7r0 = P7r.subs(initConf)
        P8r0 = P8r.subs(initConf)
        P9r0 = P9r.subs(initConf)
        PCr0 = PCr.subs(initConf)

        #endregion

        #region Link Length
        L63l = (P6l0 - P3l0).norm()
        L63r = (P6r0 - P3r0).norm()
        L64l = (P6l0 - P4l0).norm()
        L64r = (P6r0 - P4r0).norm()
        L85l = (P8l0 - P5l0).norm()
        L85r = (P8r0 - P5r0).norm()
        L71l = (P7l0 - P1l0).norm()
        L71r = (P7r0 - P1r0).norm()
        L72l = (P7l0 - P2l0).norm()
        L72r = (P7r0 - P2r0).norm()

        #endregion

        #region constraints

        # The constraint equations have the goal of keeping the length of all the links constant
        Phi2l = (P6l - P3l).dot(P6l - P3l) - L63l**2
        Phi2r = (P6r - P3r).dot(P6r - P3r) - L63r**2
        Phi3l = (P6l - P4l).dot(P6l - P4l) - L64l**2
        Phi3r = (P6r - P4r).dot(P6r - P4r) - L64r**2
        Phi4l = (P8l - P5l).dot(P8l - P5l) - L85l**2
        Phi4r = (P8r - P5r).dot(P8r - P5r) - L85r**2
        Phi5l = (P7l - P1l).dot(P7l - P1l) - L71l**2
        Phi5r = (P7r - P1r).dot(P7r - P1r) - L71r**2
        Phi6l = (P7l - P2l).dot(P7l - P2l) - L72l**2
        Phi6r = (P7r - P2r).dot(P7r - P2r) - L72r**2

        # contact points
        Phi8l = PCl[2] - self.DeltaZ
        Phi8r = PCr[2]

        # full set of equations
        Phi = sp.Matrix([Phi2l, Phi3l, Phi4l, Phi5l, Phi6l, Phi8l, Phi2r, Phi3r, Phi4r, Phi5r, Phi6r, Phi8r])
        #endregion

        # points to return
        Points = sp.Matrix.vstack(PCl.T, P1l.T, P2l.T, P3l.T, P4l.T, P5l.T, P6l.T, P7l.T, P8l.T, P9l.T)

        return Phi, Points

    def solve_kinematics(self, Phi, z_wheel):
        '''
        Given the set of equation and evaluation point returns the kinematic solution
        Phi: set of equations
        z_wheel: z heigth of the rear left wheel in respect to nominal value (m)
        '''
        eval_point = {
            self.DeltaZ : z_wheel,
            self.phi : 0.0
        }

        # setting up the problem
        vars = [self.xWl, self.xWr, self.yWl, self.yWr, self.zWl, self.zWr, self.gammaWl, self.gammaWr, self.deltaWl, self.deltaWr, self.thetaWl, self.thetaWr]
        eqns = Phi.subs(eval_point)

        # solving the problem
        sol = sp.nsolve(eqns, vars, [0]*12)

        # creating a substitution map with sol
        sol_map = dict(zip(vars, sol))
        sol_map.update(eval_point)

        return sol_map


#  _______        _   
# |__   __|      | |  
#    | | ___  ___| |_ 
#    | |/ _ \/ __| __|
#    | |  __/\__ \ |_ 
#    |_|\___||___/\__|
                     
                     

if __name__ == "__main__":
    rear_kine = RearKinematics()
    Phi, Points = rear_kine.set_kinematic_problem()
    sol = rear_kine.solve_kinematics(Phi, 0.05)
    points_sub = Points.subs(sol)
    # printing all coordinates of P2l
    #print(points_sub[2,:])


    #region Evaluation sequences
    
    # Choose kinematic intervals to study (vertical chassis motion and roll angle)
    nsteps = 21
    #Vertical wheel motion
    DeltaZmin = -0.06
    DeltaZmax = 0.06

    xWl, yWl, zWl, deltaWl, gammaWl, thetaWl = sp.symbols('xWl yWl zWl deltaWl gammaWl thetaWl')

    # Sequence of z movements in the permitted range
    dofSeqZ = np.array([[DeltaZmin + (DeltaZmax - DeltaZmin) / nsteps * i, 0] for i in range(nsteps + 1)])

    #endregion

    #region solutions

    obtainedMotionZl = np.zeros((nsteps + 1, 2), dtype=float)
    for i in range(nsteps+1):
        # Bump steer
        obtainedMotionZl[i, 0] = dofSeqZ[i, 0]
        solZ = rear_kine.solve_kinematics(Phi, dofSeqZ[i,0])
        obtainedMotionZl[i, 1] = solZ[deltaWl]*180/np.pi

    #endregion

    #region Plots

    plt.figure(figsize=(10, 5))
    plt.plot(obtainedMotionZl[:,0],obtainedMotionZl[:,1])
    plt.xlabel("dZ (m)")
    plt.ylabel("deltaWl (deg)")
    plt.title("Bumpsteer")
    plt.grid()
    plt.show()

    #endregion

