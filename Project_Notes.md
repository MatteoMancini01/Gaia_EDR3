# Key Definitions \& Project Objectives
---
### Project General Objective
The objective of this project is to use Gaia Early Data Release 3 (EDR3), to determine symmetric pattern of the solar system barycentre 
w.r.t. the rest of the Universe, this is done by measuring the proper motion of QSO-like objects. 
These objects are extremely far from us, making QSO-like objects appropriate reference rest frames. Hence the proper motion that we observe 
from data, is the solar system moving w.r.t. QSO-like object. 

### What are QSO-like objects?
A <b>Quasi-Stellar Object</b> (QSO), commonly known as a quasar, is an extremely luminous <b>Active Galactic Nucleus</b> (AGN). Most massive galaxies are believed to host a supermassive black hole at their center, with masses ranging from millions to tens of billions of solar masses. An <b>accretion disk</b> forms from gas and dust spiraling toward the black hole, heated by intense viscous and magnetic processes, emitting enormous amounts of radiation across the electromagnetic spectrum, particularly in optical, ultraviolet, and X-ray wavelengths.   

### What is aberration?
The aberration of light is an apparent shift in the position of a celestial object caused by the motion of the observer relative to the source 
of light. It is a relativistic effect, resulting from the finite speed of light and the motion of the observer.

### What is VSH (vector spherical harmonics)?
Mathematically, vector spherical harmonics are an extension of spherical harmonics (mathematical functions defined on the surface of a sphere) 
with the addition of vector fields. Complex-valued functions are the components of VSH, these are expressed as spherical coordinate basis vectors. 
For the purposes of this project, VSH will be used as the model distribution to fit the data from Gaia ERD3.

### What is the Gaia mission?
Gaia is a space observatory launched in 2013 by the European Space Agency (ESA). 
The spacecraft is designed to collect data from space in particular, measurements of positions, distances and motion of stars. 
We aim to construct the largest 3D space catalog ever made, containing approximately over 1 billion astronomical objects, also including planets, 
comets, asteroids and quasars (these are vital for the project purpose).

### What is VLBI?
Very-long-baseline interferometry (VLBI) is a type of astronomical interferometry used in used in radio astronomy. 
This is also used to collect data on quasars, the main difference to Gaia is that VLBI works in the radio wavelength 
and has been historically used to define the celestial reference frame. While Gaia, on the other hand, 
operates in optical wavelengths and has defined the Gaia Celestial Reference Frame (Gaia-CRF), complementing the ICRF.

### Baryons

Baryons are particles like protons and neutrons. A baryonic component refers to the part of a system, such as galaxies, galactic clusters or the universe itself, that is made of  baryonic matter. Examples include, starts, gas, dust, planets, including black holes and neutron stars if they are from baryonic matter. Essentially, it includes all the "normal" matter we are familiar with.

### Key concepts on Vector Spherical Harmonics (VSH):

1. Definition and Basics of Vector Spherical Harmonics (VSH):
   - Understanding the mathematical framework of VSH.
   - Difference between scalar spherical hamonics and VSH.

2. Application of VSH in Astronomy
   - How VSH is used to analyse vector fields in celestial mechanics
   - The role of VSH in decomposing proper motion fields

3. VSH Expansion and Representation
   - Representation of vector fields in celestial mechanics.
   - Decomposition into toroidal and spheroidal components. 
   - The significance of first-order harmonics ($l = 1$) for detecting systematic effects.

4. Acceleration Effects in VSH
   - How acceleration of the Solar System is modeled using VSH.
   - Relationship between VSH expansion and dipole pattern of acceleration.
   - Estimating the length of an acceleration vector from its Cartesian components.

5. Numerical and Computational Aspect of VSH
   - Least-square fitting techinques used in VSH analysis.
   - Handing systematic biases in the expectation.
   - Methods for ensuring robustness against outliers.

6. VSH in the Gaia Mission
   - Application of VSH to analyse Gaia data.
   - Detecting and quantifying the acceleration of the Solar System.
   - Handling large-scale systematics in Gaia astrometric solutions.

7. Error Analysis and Bias Considerations
   - Understanding transformation biases in spherical coordinate system.
   - Analysing uncertainties and correlations in VSH expansion.
   - Bootstrap resampling methods for error estimation.

8. Comparison with Other Techniques
   - Difference between direct least-squares fitting of acceleration components and VSH expansion.
   - Benefits of using VSH for mitigating systematic errors in proper motion analysis. 

---

# Project Log

This section aims to keep track of the progress made with the project. I will report all the updates made before pushing to the remote repository. This aims to facilitate writing the report once the project is completed.

---

### Downloading Data

Follow the instructions presented in the [README.md](README.md#downloading-data-ringed_planet-telescope-floppy_disk) file