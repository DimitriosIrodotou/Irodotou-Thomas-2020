import numpy as np
import healpy as hlp
import astropy.units as u
from astropy_healpix import HEALPix


def decomposition_IT20(sp_mass, sp_am_unit_vector):
    """
    Perform disc/spheroid decomposition of simulated galaxies based on the method introduced in Irodotou and Thomas 2020 (hereafter IT20)
    (https://ui.adsabs.harvard.edu/abs/2020arXiv200908483I/abstract).
    This function takes as arguments the mass and angular momentum of stellar particles and returns two masks that contain disc and spheroid particles
    and the disc-to-total stellar mass ratio of the galaxy.

    # Step (i) in Section 2.2 Decomposition of IT20 #
    :param sp_mass: list of stellar particles's (sp) masses.
    :param sp_am_unit_vector: list of stellar particles's (sp) normalised angular momenta (am) unit vectors.
    :return: disc_mask_IT20, spheroid_mask_IT20, disc_fraction_IT20
    """

    # Step (ii) in Section 2.2 Decomposition of IT20 #
    # Calculate the azimuth (alpha) and elevation (delta) angle of the angular momentum of all stellar particles #
    alpha = np.degrees(np.arctan2(sp_am_unit_vector[:, 1], sp_am_unit_vector[:, 0]))  # In degrees.
    delta = np.degrees(np.arcsin(sp_am_unit_vector[:, 2]))  # In degrees.

    # Step (ii) in Section 2.2 Decomposition of IT20 #
    # Generate the pixelisation of the angular momentum map #
    nside = 2 ** 4  # Define the resolution of the grid (number of divisions along the side of a base-resolution grid cell).
    hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
    indices = hp.lonlat_to_healpix(alpha * u.deg, delta * u.deg)  # Create a list of HEALPix indices from particles's alpha and delta.
    densities = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix grid cell.

    # Step (iii) in Section 2.2 Decomposition of IT20 #
    # Smooth the angular momentum map with a top-hat filter of angular radius 30 degrees #
    smoothed_densities = np.zeros(hp.npix)
    # Loop over all grid cells #
    for i in range(hp.npix):
        mask = hlp.query_disc(nside, hlp.pix2vec(nside, i), np.pi / 6.0)  # Do a 30 degree cone search around each grid cell.
        smoothed_densities[i] = np.mean(densities[mask])  # Average the densities of the ones inside and assign this value to the grid cell.

    # Step (iii) in Section 2.2 Decomposition of IT20 #
    # Find the location of the density maximum #
    index_densest = np.argmax(smoothed_densities)
    alpha_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi  # In radians.
    delta_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2  # In radians.

    # Step (iv) in Section 2.2 Decomposition of IT20 #
    # Calculate the angular separation of each stellar particle from the centre of the densest grid cell #
    Delta_theta = np.arccos(np.sin(delta_densest) * np.sin(np.radians(delta)) + np.cos(delta_densest) * np.cos(np.radians(delta)) * np.cos(
        alpha_densest - np.radians(alpha)))  # In radians.

    # Step (v) in Section 2.2 Decomposition of IT20 #
    # Calculate the disc mass fraction as the mass within 30 degrees from the densest grid cell #
    disc_mask_IT20, = np.where(Delta_theta < (np.pi / 6.0))
    spheroid_mask_IT20, = np.where(Delta_theta >= (np.pi / 6.0))
    disc_fraction_IT20 = np.sum(sp_mass[disc_mask_IT20]) / np.sum(sp_mass)

    # Step (vi) in Section 2.2 Decomposition of IT20 #
    # Normalise the disc fractions #
    chi = 0.5 * (1 - np.cos(np.pi / 6))
    disc_fraction_IT20 = np.divide(1, 1 - chi) * (disc_fraction_IT20 - chi)

    return disc_mask_IT20, spheroid_mask_IT20, disc_fraction_IT20
