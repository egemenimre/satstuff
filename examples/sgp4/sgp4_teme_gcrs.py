"""
Sample `astropy` conversion between TEME and GCRS to be able to use SGP4 trajectory output.

Licensed under GNU GPL v3.0. See LICENSE.rst for more info.

"""
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, CartesianDifferential, GCRS
from astropy.time import Time, TimeDelta
from sgp4.model import Satrec

from frames.teme import TEME

if __name__ == '__main__':
    # ****** Config and general setting up ******
    # Example is from the book:
    # Fundamentals of Astrodynamics and Applications 4th Ed. David A. Vallado, Section 3.7, pp.233-234

    # Init TLE
    line1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
    line2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"

    # Init satellite object from the TLE
    sat = Satrec.twoline2rv(line1, line2)

    # Define output time
    output_time: Time = Time("2000:182", scale="utc", format="yday") + TimeDelta(0.78495062, format="jd")
    output_time.format = "iso"

    # SGP4 module requires time instances as jd and fraction
    # This is compatible with Astropy time class
    jd = output_time.jd1
    frac = output_time.jd2

    # Run the propagation and init pos and vel vectors in TEME
    e, r_teme, v_teme = sat.sgp4(jd, frac)

    # Load the time, pos, vel info into astropy objects
    coords = CartesianRepresentation(x=r_teme, unit=u.km, copy=True) \
        .with_differentials(CartesianDifferential(d_x=v_teme, unit=u.km / u.s, copy=True))

    print(f"Time              : {output_time.iso}")
    print(f"Pos vector (TEME) : {coords}")
    print(f"Vel vector (TEME) : {coords.differentials['s']}")

    # Load coordinates into TEME object
    coord_teme = TEME(coords, obstime=output_time, representation_type="cartesian", differential_type="cartesian")

    print(f"TEME object: {coord_teme}")

    # Vallado pg.234
    v_TEME_true = CartesianDifferential(d_x=[-2.232832783, -4.110453490, -3.157345433], unit=u.km / u.s, copy=True)
    r_TEME_true = CartesianRepresentation(x=[-9060.47373569, 4658.70952502, 813.68673153], unit=u.km, copy=True)
    r_TEME_true = TEME(r_TEME_true.with_differentials(v_TEME_true), obstime=output_time,
                       representation_type="cartesian",
                       differential_type="cartesian")
    # check the SGP4 results
    print(
        f"r TEME diff      :  {(coord_teme.cartesian.without_differentials() - r_TEME_true.cartesian.without_differentials()).norm().to(u.mm)}")
    print(f"v TEME diff      :  {(coord_teme.velocity - r_TEME_true.velocity).norm().to(u.mm / u.s)}")

    # Convert to GCRS
    coord_gcrs = coord_teme.transform_to(GCRS(obstime=output_time))

    print(f"Pos vector (GCRS) : {coord_gcrs.cartesian}")
    print(f"Vel vector (GCRS) : {coord_gcrs.velocity}")

    # Vallado pg.234 - this is actually in J2000 but the difference is less than a meter
    v_GCRS_true = CartesianDifferential(d_x=[-2.233348094, -4.110136162, -3.157394074], unit=u.km / u.s, copy=True)
    r_GCRS_true = CartesianRepresentation(x=[-9059.9413786, 4659.6972000, 813.9588875], unit=u.km, copy=True)
    r_GCRS_true = GCRS(r_GCRS_true.with_differentials(v_GCRS_true), obstime=output_time,
                       representation_type="cartesian",
                       differential_type="cartesian")

    # check the coord conversion results
    print(
        f"r GCRS diff      :  {(coord_gcrs.cartesian.without_differentials() - r_GCRS_true.cartesian.without_differentials()).norm().to(u.mm)}")
    print(f"v GCRS diff      :  {(coord_gcrs.velocity - r_GCRS_true.velocity).norm().to(u.mm / u.s)}")
