"""
Use `astropy` to generate Sun and Moon rise-set times for a location on Earth.

Licensed under GNU GPL v3.0. See LICENSE.rst for more info.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytz

from astropy import units as u
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
from astropy.coordinates import get_sun, get_moon
from astropy.time import Time, TimeDelta

if __name__ == '__main__':

    # Init location - default Ellipsoid is WGS84
    istanbul: EarthLocation = EarthLocation(lat=41.015137, lon=28.979530, height=0 * u.m)
    ist_timezone = pytz.timezone("Turkey")

    # Time analysis config (stepsize, duration, init time)
    init_time: Time = Time("2020-04-12T00:00:00", scale="utc")
    dt = TimeDelta(10 * 60.0, format="sec")  # stepsize
    duration = TimeDelta(1.0, format="jd")  # duration

    # Generate observation time list
    dt_list = dt * np.arange(0, duration.sec / dt.sec, 1)
    obs_times: Time = init_time + dt_list

    # Init the frames for each time
    alt_az_frames: AltAz = AltAz(location=istanbul, obstime=obs_times)

    # Generate the Sun coords in Alt Az
    sun_alt_az_list = get_sun(obs_times).transform_to(alt_az_frames)
    moon_alt_az_list = get_moon(obs_times).transform_to(alt_az_frames)

    # ***** Plot the Sun and Moon Alt angles *****
    time_list = obs_times.to_datetime(timezone=ist_timezone)
    plt.plot(time_list, sun_alt_az_list.alt.deg, time_list, moon_alt_az_list.alt.deg)
    plt.grid()

    # Plot a min elevation angle limit
    min_elev = 5  # deg
    plt.axhline(min_elev, color="k", ls="dashed")

    # autoformat the time labels on x-axis
    # plt.gcf().autofmt_xdate()
    plt.xticks(rotation=90)

    plt.title(f"Sun and Moon Altitude at Istanbul ({init_time.to_datetime(timezone=ist_timezone).date()})")
    plt.xlabel("Time [Local]")
    plt.ylabel("Altitude [deg]")
    plt.legend(["Sun", "Moon", f"{min_elev} deg el"])

    plt.show()
