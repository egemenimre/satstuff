"""
Use `astropy` and an interpolator to generate Sun rise-set times for a location on Earth.

Licensed under GNU GPL v3.0. See LICENSE.rst for more info.
"""
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
from astropy.coordinates import get_sun
from astropy.time import Time, TimeDelta
from scipy import interpolate

if __name__ == '__main__':

    # Init location - default Ellipsoid is WGS84
    istanbul: EarthLocation = EarthLocation(lat=41.015137, lon=28.979530, height=0 * u.m)
    ist_timezone = pytz.timezone("Turkey")
    print(f"Target coordinates (lat,lon): {istanbul.lat}, {istanbul.lon}")

    # Time analysis config (stepsize, duration, init time)
    init_time: Time = Time("2020-04-12T00:00:00", scale="utc")
    dt = TimeDelta(10 * 60.0, format="sec")  # stepsize
    duration = TimeDelta(1.0, format="jd")  # duration

    # Generate observation time list
    dt_list = dt * np.arange(0, duration.sec / dt.sec, 1)
    obs_times: Time = init_time + dt_list

    # Init the frames for each time
    alt_az_frames: AltAz = AltAz(location=istanbul, obstime=obs_times)

    # Generate the Sun coords (low precision apparent pos) in Alt Az
    sun_alt_az_list = get_sun(obs_times).transform_to(alt_az_frames)

    # ***** Generate the altitude interpolators *****

    # Convert time list into a float list for use with the interpolator (possible loss of precision)
    t__float_list = list(map(lambda t: t.timestamp(), sun_alt_az_list.obstime.to_datetime()))

    # Init interpolators (for splines, root finding possible for 3rd degree (cubics) only)
    sun_alt_interpolator = interpolate.Akima1DInterpolator(t__float_list, sun_alt_az_list.alt.deg)
    sun_interpolator_deriv_alt = sun_alt_interpolator.derivative()

    # Sample interpolation result - to check things
    interp_time: Time = Time("2020-04-12 03:33:48.776906", scale="utc")
    sun_interp_alt = sun_alt_interpolator(interp_time.to_datetime().timestamp())

    print(f"Sample interpolation result at {interp_time.to_datetime(ist_timezone)}")
    print(
        f"Sun true alt (deg)        : {get_sun(interp_time).transform_to(AltAz(location=istanbul, obstime=interp_time)).alt.deg}")
    print(f"Sun interpolated alt (deg): {sun_interp_alt}")

    print(" --------------------- ")

    # Find the events and characteristics
    sun_rise_time: Time
    sun_set_time: Time
    rise_set_events = sun_alt_interpolator.roots()
    for event_time in rise_set_events:
        deriv = sun_interpolator_deriv_alt(event_time)
        if deriv > 0:
            sun_rise_time = Time(datetime.fromtimestamp(event_time), format="datetime", scale="utc")
            print("Sunrise time:")
        else:
            sun_set_time = Time(datetime.fromtimestamp(event_time), format="datetime", scale="utc")
            print("Sunset time:")
        print(
            Time(datetime.fromtimestamp(event_time), format="datetime", scale="utc").to_datetime(timezone=ist_timezone))
        # print(f"Altitude (deg) : {sun_alt_interpolator(event_time)}")

    # Akima
    # 2020-04-12 03:33:48.766705
    # 2020-04-12 16:36:27.265486
    # 2020-04-12 10:04:56.620527
    # 2020-04-12 22:04:24.888192

    # InterpolatedUnivariateSpline
    # 2020-04-12 03:33:48.776906
    # 2020-04-12 16:36:27.253373
    # 2020-04-12 10:04:56.592995
    # 2020-04-12 22:04:24.787477

    min_max_events = sun_interpolator_deriv_alt.roots()
    for event_time in min_max_events:
        pos = sun_alt_interpolator(event_time)
        if pos > 0:
            print("Sun highest time:")
        else:
            print("Sun lowest time:")
        print(
            Time(datetime.fromtimestamp(event_time), format="datetime", scale="utc").to_datetime(timezone=ist_timezone))
        print(f"Altitude (deg) : {sun_alt_interpolator(event_time)}")

    # ***** Plot the Sun Alt angles *****
    time_list = obs_times.to_datetime(timezone=ist_timezone)
    plt.plot(time_list, sun_alt_az_list.alt.deg)
    plt.grid()

    # Add sunrise and sunset markers
    plt.axvline(sun_rise_time.to_datetime(timezone=ist_timezone), color="red", ls="dashed")
    plt.axvline(sun_set_time.to_datetime(timezone=ist_timezone), color="green", ls="dashed")

    # autoformat the time labels on x-axis
    # plt.gcf().autofmt_xdate()
    plt.xticks(rotation=90)

    plt.title(f"Sun Altitude at Istanbul ({init_time.to_datetime(timezone=ist_timezone).date()})")
    plt.xlabel("Time [Local]")
    plt.ylabel("Altitude [deg]")
    plt.legend(["Sun", "sun rise", "sun set"])

    plt.show()
