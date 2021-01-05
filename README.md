Forward Model v0 : 

Source is a dielectric monopole in the middle of the cavity
Metasurface is a PEC/PMC distribution
Aperture is a homogeneous dielectric layer with impedance layer for transmittance modulation
Field are given on the aperture

Folders: 
- Aperture: contains the functions realitive to creating the aperture layer reflection matrices
- Cavity: contains the functions realitive to  symmetrization of fields and configuration of the metasurface
- Fourier Green: contains the functions realitive to creating the dyadic Green's functions in Fourier domain and propagation
- Pytorch Functions: contains the previous functions converted to pytorch formalism

Python codes: 
- ForwardModelPython runs a Python simulation for this version of the model using numpy arrays
- ForwardModelPytorch runs a Python simulation for this version of the model using Tensor objects
- ForwardModelPytorchforNN runs a Python simulation for this version of the model using Tensor objects (forward and backward model)