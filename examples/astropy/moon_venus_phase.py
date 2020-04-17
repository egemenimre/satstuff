"""
Phase angle of the Moon and Venus using `astropy`.

Licensed under GNU GPL v3.0. See LICENSE.rst for more info.

"""

import matplotlib.pyplot as plt
import numpy as np
import pytz

from astropy import units as u
from astropy.coordinates import EarthLocation, get_moon, get_sun, SkyCoord, CartesianRepresentation, get_body
from astropy.time import Time, TimeDelta

if __name__ == '__main__':
    # Init location - default Ellipsoid is WGS84
    istanbul: EarthLocation = EarthLocation(lat=41.015137, lon=28.979530, height=0 * u.m)
    ist_timezone = pytz.timezone("Turkey")
    utc_timezone = pytz.timezone("utc")
    print(f"Target coordinates (lat,lon): {istanbul.lat}, {istanbul.lon}")

    # Time analysis config (stepsize, duration, init time)
    init_time: Time = Time("2020-04-01T00:00:00", scale="utc")
    dt = TimeDelta(0.1, format="jd")  # stepsize
    duration = TimeDelta(90.0, format="jd")  # duration

    # Generate observation time list
    dt_list = dt * np.arange(0, duration.sec / dt.sec, 1)
    obs_times: Time = init_time + dt_list

    # Generate Sun, Moon and Venus coordinates
    sun_vec_gcrs: SkyCoord = get_sun(obs_times).cartesian
    moon_vec_gcrs: SkyCoord = get_moon(obs_times).cartesian
    venus_vec_gcrs: SkyCoord = get_body("venus", obs_times).cartesian

    # Generate Earth location in GCRS
    gnd_loc_gcrs = istanbul.get_gcrs(obs_times).cartesian.without_differentials()

    # Generate Sun, Moon and Venus-to-Istanbul vectors
    sun_to_moon: CartesianRepresentation = sun_vec_gcrs - moon_vec_gcrs
    gnd_to_moon: CartesianRepresentation = gnd_loc_gcrs - moon_vec_gcrs

    sun_to_venus: CartesianRepresentation = sun_vec_gcrs - venus_vec_gcrs
    gnd_to_venus: CartesianRepresentation = gnd_loc_gcrs - venus_vec_gcrs

    # Compute angle between two vectors for each instant in time
    sun_to_moon_unit = sun_to_moon / sun_to_moon.norm()
    gnd_to_moon_unit = gnd_to_moon / gnd_to_moon.norm()
    phase_angle_moon = np.rad2deg(np.arccos(sun_to_moon_unit.dot(gnd_to_moon_unit)))

    sun_to_venus_unit = sun_to_venus / sun_to_venus.norm()
    gnd_to_venus_unit = gnd_to_venus / gnd_to_venus.norm()
    phase_angle_venus = np.rad2deg(np.arccos(sun_to_venus_unit.dot(gnd_to_venus_unit)))

    print(f"Moon phase angle  : {phase_angle_moon[0]} at {obs_times[0].to_datetime(timezone=utc_timezone)}")
    print(f"Venus phase angle : {phase_angle_venus[0]} at {obs_times[0].to_datetime(timezone=utc_timezone)}")

    # Find an approx full moon
    full_moon_index_search, = np.where(phase_angle_moon == phase_angle_moon.min())
    full_moon_index = full_moon_index_search[0]  # numpy.where outputs a tuple, use first element
    full_moon_time = obs_times[full_moon_index]
    print(
        f"Approx full moon  : {phase_angle_moon[full_moon_index]} at {full_moon_time.to_datetime(timezone=utc_timezone)}")

    # ***** Plot the Moon and Venus illumination percentages *****
    time_list = obs_times.to_datetime(timezone=ist_timezone)
    plt.plot(time_list, phase_angle_moon, time_list, phase_angle_venus)
    plt.grid()

    # Add full Moon marker
    plt.axvline(full_moon_time.to_datetime(timezone=ist_timezone), color="green", ls="dashed")

    # autoformat the time labels on x-axis
    plt.gcf().autofmt_xdate()
    # plt.xticks(rotation=90)

    plt.title(f"Moon and Venus Phase Angles as seen from Istanbul "
              f"\n({obs_times[0].to_datetime(timezone=ist_timezone).date()}"
              f" - {obs_times[-1].to_datetime(timezone=ist_timezone).date()})")
    plt.xlabel("Time [Local]")
    plt.ylabel("Phase Angle [deg]")
    plt.legend(["Moon", "Venus", "Approx Full Moon"])

    plt.show()
