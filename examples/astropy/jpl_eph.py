"""
Example for basic JPL Ephemerides and default ephemerides use.

Truth values from JPL Horizons Website https://ssd.jpl.nasa.gov/?horizons.

Licensed under GNU GPL v3.0. See LICENSE.rst for more info.
"""
import numpy as np

from astropy import units as u
from astropy.coordinates import solar_system_ephemeris, get_body, get_sun
from astropy.time import Time

if __name__ == '__main__':
    # Target time
    init_time: Time = Time("2020-04-12T00:00:00", scale="tdb")

    print(f"Target time in {init_time.scale}: {init_time}")

    print("-------------------")
    print("Sun coords with low precision default:")
    with solar_system_ephemeris.set("builtin"):
        sun_builtin = get_body('sun', init_time)
    print(sun_builtin.cartesian.xyz.to(u.km))

    print("-------------------")
    print("Sun coords with get_sun:")
    sun_getsun = get_sun(init_time)
    print(sun_getsun.cartesian.xyz.to(u.km))

    print("-------------------")
    print("Sun coords with high precision de432s:")
    with solar_system_ephemeris.set("de432s"):
        sun_jpl = get_body('sun', init_time)
    print(sun_jpl.cartesian)

    # Get the "truth" data from JPL Horizons Website

    # Sun geometric coordinates
    # -------------------------
    X = 1.387689430984850E+08
    Y = 5.215091267413133E+07
    Z = 2.260706819574005E+07
    # VX = -1.079363638048313E+01
    # VY = 2.539120848257597E+01
    # VZ = 1.100591845217031E+01

    sun_de431mx_geo_truth = np.array([X, Y, Z]) * u.km

    # Sun coordinates with light-time correction
    # -------------------------
    X = 1.387689501964446E+08
    Y = 5.215091512353201E+07
    Z = 2.260706903949238E+07
    # VX = -1.079363638972499E+01
    # VY = 2.539120858607189E+01
    # VZ = 1.100591849667028E+01

    sun_de431mx_lt_corr_truth = np.array([X, Y, Z]) * u.km

    # Sun coordinates with light-time and stellar correction
    # -------------------------
    X = 1.387745804886835E+08
    Y = 5.213830132809754E+07
    Z = 2.260160176208604E+07
    # VX = -1.079363638972499E+01
    # VY = 2.539120858607189E+01
    # VZ = 1.100591849667028E+01

    print("Difference (Light time and Stellar aberration corrected values):")
    sun_de431mx_lt_s_corr_truth = np.array([X, Y, Z]) * u.km
    print(sun_jpl.cartesian.xyz.to(u.km) - sun_de431mx_lt_s_corr_truth)
