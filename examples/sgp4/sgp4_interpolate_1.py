"""
This is a an example for Brandon Rhodes' `sgp4` module with the trajectory interpolation thrown in.

Licensed under GNU GPL v3.0. See LICENSE.rst for more info.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from sgp4.api import Satrec
# from sgp4.api import SGP4_ERRORS
from sgp4.api import accelerated

if __name__ == '__main__':
    # np.set_printoptions(precision=2)

    # Init TLE
    line1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
    line2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"

    # Init satellite object from the TLE
    sat = Satrec.twoline2rv(line1, line2)

    # jd_init, fr = jday(2019, 12, 9, 12, 0, 0)

    # ****** Generate the discrete time instants through the propagation duration ******
    jd_init = 2458826
    init_time = 0.0  # days
    end_time = 3.0  # days
    steps = 4000  # number of steps between init and end
    stepsize = (end_time - init_time) / steps  # stepsize in days
    jd_list = np.full(steps, jd_init)
    fr_list = np.arange(init_time, end_time, (end_time - init_time) / steps)

    print(f"Propagation duration: {end_time - init_time} days")
    print(f"Step size: {stepsize * 86400} sec")

    # ****** Generate the pos, vel vectors for each time instant ******
    timer_start = time.perf_counter()  # start timer for performance check

    r_truth_list: np.ndarray
    e, r_truth_list, v_truth_list = sat.sgp4_array(jd_list, fr_list)

    timer_end = time.perf_counter()  # end timer for performance check
    execution_time_in_sec = timer_end - timer_start

    print(f"Is accelerated: {accelerated} \n")

    # Run duration for 10000 runs = 350 msec in run mode
    print(f"Took {execution_time_in_sec * 1000 : 06.6f} milliseconds to generate {steps} (t,r,v) combinations.")

    # # Output a sample step
    # index = steps - 1
    # print(f"Error code: {e[index]}")
    # print(f"Time (JD): {jd_list[index]} + {fr_list[index]}")
    # print(f"TEME Position (km):   {r[index]}")
    # print(f"TEME Velocity (km/s): {v[index]}")

    # ****** Fill interpolator with time and (x, y, z) axes ******
    spline_degree = 5  # degree of spline
    extrapolate_action = "raise"  # raise Exception if an out of bounds point is requested

    timer_start = time.perf_counter()  # start timer for performance check

    r_x_interpol = interpolate.InterpolatedUnivariateSpline(fr_list, r_truth_list[:, 0], k=spline_degree,
                                                            ext=extrapolate_action)
    r_y_interpol = interpolate.InterpolatedUnivariateSpline(fr_list, r_truth_list[:, 1], k=spline_degree,
                                                            ext=extrapolate_action)
    r_z_interpol = interpolate.InterpolatedUnivariateSpline(fr_list, r_truth_list[:, 2], k=spline_degree,
                                                            ext=extrapolate_action)

    timer_end = time.perf_counter()  # end timer for performance check
    execution_time_in_sec = timer_end - timer_start

    print(f"Took {execution_time_in_sec * 1000 : 06.6f} milliseconds to generate x, y, z interpolators "
          f"for {len(fr_list)} data points.")

    # ******  Test interpolation accuracy ******
    step_offset = 0
    test_stepsize = 1.0 / 86400  # stepsize in days
    test_end = 0.01  # days

    print(f"Test duration: {test_end - step_offset * stepsize} days")
    print(f"Step size: {test_stepsize * 86400} sec")

    fr_test_list = np.arange(step_offset * stepsize, test_end, test_stepsize)
    jd_test_list = np.full(len(fr_test_list), jd_init)

    # Generate the high-res test trajectory
    e_test_list, r_test_list, v_test_list = sat.sgp4_array(jd_test_list, fr_test_list)
    r_err_list = np.array([r_x_interpol(fr_test_list), r_y_interpol(fr_test_list), r_z_interpol(fr_test_list)])
    r_err_list = r_err_list.transpose() - r_test_list

    # t_tgt = step_offset * stepsize + 21 / 86400
    # err_code, r_truth, v_truth = sat.sgp4(jd_list[0], t_tgt)  # True coords for this time instant
    # r_ipol = np.array([r_x_interpol(t_tgt), r_y_interpol(t_tgt), r_z_interpol(t_tgt)])  # Interpolation

    # print(f"Target time: {t_tgt} days from epoch")
    # print(f"True coords (km)         : {r_truth}")
    # print(f"Interpolated coords (km) : {r_ipol}")
    # print(f"Difference (m) : {(r_ipol - r_truth) * 1000}")

    # Plot the error per axis
    fr_ticker = np.arange(fr_test_list[0], fr_test_list[-1], stepsize)
    y_ticker = np.zeros(len(fr_ticker))
    plt.figure()
    # plt.xkcd()  # useless but fun xkcd format
    plt.plot(fr_test_list, r_err_list, fr_ticker, y_ticker, "x")
    plt.legend(["x", "y", "z", "stepsize"])
    plt.title(f"Interpolation Error\n({r_x_interpol.__class__.__name__} @ Degree: {spline_degree})")
    plt.xlabel("Time [days]")
    plt.ylabel("Error [m]")
    # plt.yscale("log")  # Uncomment to get y axis in log scale
    plt.show()
