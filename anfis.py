"""Script demonstrating ANFIS control for quadcopter.

The simulation is run by a `CtrlAviary` environment.
The control is given by the ANFIS implementation in `ANFISControl`.

Example
-------
In a terminal, run as:

    $ python anfis_control.py

Parameters
----------
--drone : DroneModel
    Type of drone model to use (default: CF2X)
--num_drones : int
    Number of drones in the simulation (default: 1)
--physics : Physics
    Physics engine to use (default: PyBullet)
--gui : bool
    Whether to use PyBullet GUI (default: True)
--record_video : bool
    Whether to record video of the simulation (default: False)
--plot : bool
    Whether to plot the simulation results (default: True)
--user_debug_gui : bool
    Whether to add debug lines and parameters to GUI (default: False)
--obstacles : bool
    Whether to add obstacles to the environment (default: True)
--simulation_freq_hz : int
    Simulation frequency in Hz (default: 240)
--control_freq_hz : int
    Control frequency in Hz (default: 48)
--duration_sec : int
    Duration of the simulation in seconds (default: 30)
--output_folder : str
    Folder where to save logs (default: 'anfis_results')
--trajectory : str
    Trajectory to follow (default: 'circle')
--num_mfs : int
    Number of membership functions per input (default: 3)
--learning_rate : float
    Learning rate for ANFIS adaptation (default: 0.01)
"""
import os
import time
import argparse
from datetime import datetime
import math
import numpy as np
import pybullet as p


from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from ANFISControl import ANFISControl
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2p")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_DURATION_SEC = 180
DEFAULT_OUTPUT_FOLDER = 'anfis_results'
DEFAULT_COLAB = False
DEFAULT_TRAJECTORY = "circle"  # "circle", "figure8", "hover"
DEFAULT_NUM_MFS = 3
DEFAULT_LEARNING_RATE = 1e-5

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        trajectory=DEFAULT_TRAJECTORY,
        num_mfs=DEFAULT_NUM_MFS,
        learning_rate=DEFAULT_LEARNING_RATE
        ):
    
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))

    #### Initialize trajectory ######################
    if trajectory == "circle":
        for i in range(NUM_WP):
            TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    elif trajectory == "figure8":
        a = R  # radius of the figure 8
        for i in range(NUM_WP):
            t = (i/NUM_WP)*2*np.pi
            x = a*np.sin(t)/(1+np.cos(t)**2)
            y = a*np.sin(t)*np.cos(t)/(1+np.cos(t)**2)
            TARGET_POS[i, :] = x+INIT_XYZS[0, 0], y+INIT_XYZS[0, 1], 0
    elif trajectory == "hover":
        # Simple hover trajectory
        for i in range(NUM_WP):
            TARGET_POS[i, :] = INIT_XYZS[0, 0], INIT_XYZS[0, 1], 0

    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=100,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=False,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the ANFIS controllers ######################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [ANFISControl(drone_model=drone, 
                            num_mfs=num_mfs, 
                            learning_rate=learning_rate) for i in range(num_drones)]
        
        pid_teacher_ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
        print(f"ANFIS Controller initialized:")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Trajectory: {trajectory}")
    else:
        print("[ERROR] ANFIS control requires CF2X or CF2P drone model")
        exit()

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))
    START = time.time()

    
    print(f"Starting ANFIS control simulation...")
    print(f"   - Duration: {duration_sec} seconds")
    print(f"   - Control frequency: {control_freq_hz} Hz")
    
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            ideal_rpm, _, _ = pid_teacher_ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                target_rpy=INIT_RPYS[j, :]
            )

            action[j, :], pos_e, yaw_e, anfis_intermediate_outputs = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                target_rpy=INIT_RPYS[j, :]
            )

            ideal_thrust_pwm, ideal_torques = pid_teacher_ctrl[j].getLastThrustAndTorques()
            
            ctrl[j].learn(
                ideal_thrust_pwm, 
                ideal_torques,
                anfis_intermediate_outputs
            )

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Print progress ########################################
        if i % (env.CTRL_FREQ * 5) == 0:  # Print every 5 seconds
            print(f"   ⏱️  Simulation time: {i/env.CTRL_FREQ:.1f}s / {duration_sec}s")

        #### Render the simulation #################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    print("Saving simulation results...")
    logger.save()
    logger.save_as_csv("anfis")

    #### Plot the simulation results ###########################
    if plot:
        print("Plotting results...")
        logger.plot()
        

if __name__ == "__main__":
    #### Define and parse arguments ############################
    parser = argparse.ArgumentParser(description='ANFIS control script using CtrlAviary and ANFISControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 1)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 30)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "anfis_results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--trajectory',          default=DEFAULT_TRAJECTORY, type=str,           help='Trajectory to follow (default: "circle")', metavar='', choices=["circle", "figure8", "hover"])
    parser.add_argument('--num_mfs',            default=DEFAULT_NUM_MFS, type=int,           help='Number of membership functions per input (default: 3)', metavar='')
    parser.add_argument('--learning_rate',      default=DEFAULT_LEARNING_RATE, type=float,         help='Learning rate for ANFIS adaptation (default: 0.01)', metavar='')
    
    ARGS = parser.parse_args()

    run(**vars(ARGS))