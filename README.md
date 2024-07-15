# EF-TIGRE: Effective Field Theory of Interacting dark energy with Gravitational REdshift

Package for constraining interacting Dark Energy/Dark Matter models in the Effective Field Theory framework throuhgh Large Scale Structures observables. In particular, the observables include the effect of gravitational redshift - a distortion of time from galaxy clustering. This generate a dipole in the correlation function, detectable with two distinct populations of galaxies. This makes it possible to break degeneracies among parameters of the EFT description.

For the formulation of the Effective Field Theory of Interacting Dark Energy, see [arXiv:1504.05481](<https://arxiv.org/abs/1504.05481>) and [arXiv:1509.02191](<https://arxiv.org/abs/1509.02191>).

For the effects of gravitational redshift see [arXiv:1309.1321](<https://arxiv.org/abs/1309.1321>), [arXiv:1304.4124](<https://arxiv.org/abs/1304.4124>) and [arXiv:1206.5809](<https://arxiv.org/abs/1206.5809>).

For a study of the constraining power of gravitational redshift on the Effective Field Theory of Interacting Dark Energy, see [arXiv:2311.14425](<https://arxiv.org/abs/2311.14425>) for which this package has been developed.

Developed by [Michele Mancarella](<https://github.com/Mik3M4n>), [Sveva Castello](<https://github.com/SvevaCastello>), [Nastassia Grimm](<https://github.com/NastassiaG>), [Daniel Sobral Blanco](<https://github.com/dasobral>).




## Code Organization
The organisation of the repository is the following:

```
EFTfunctions.py 
	Parametrization of the time evolution for the EFT free functions

background.py 
	Logic for solving the background evolution and computing background quantities
	
perturbations.py  
	Logic for solving the evolution of perturbations

multipole_signal.py  
	Computation of the multipoles
	
likelihood_tools
	Likelihood, priors, posteriors			
						
```

## Usage

For a more detailed description see the example in  ```notebooks/example.ipynb```. 

Quickstart: background evolution

```python

from background import  BackgroundGammac
from EFTfunctions import  EFTfunGammac

aM = 0.1 # running of Planck mass
aB = -0.5 # kinetic braiding
gamma_par = 0.1 # DE-DM coupling
w  =-1.1 # DE Equation of state


# Instantiate EFT object. Contains time evolution for the free functions
EFT = EFTfunGammac(aM, aB, gamma_par, tol=0.)

# Instantiate background object
BG = BackgroundGammac( 0.31,  # matter energy density today
							0.05, # baryon energy density today
							w,  
							EFT, 
							zin=1100, # initial redshift for bg solution							tol=1e-15, 
							Or0=9e-05, # radiation energy density today
							xin_cs=-np.log(11) # value of x=ln(a) after which we impose stability of the perturbations
							)
    
    
# Solve for background evolution
    BG.solve( method='DOP853', res=100, rtol=1e-08, atol=1e-15, verbose=True)
    
# check stability
if not BG.is_stable():  
     print('Unstable model! Check parameters')
else:
     print('Stable model.')

```

Quickstart: perturbations evolution

```python

from perturbations import Perturbations

myPert = Perturbations(BG, 
							zin=10, # initial redshift from which we evolve perturbations
							tol=1e-15, 
							X=1.  # fraction of DM within a galaxy inunits of the background fraction of DM
							)
    
myPert.solve(method='DOP853', res=100, rtol=1e-05, atol=1e-10, verbose=True)

```


Quickstart: compute the signal

```python

from multipole_signal import Signal
from mul_nu1 import interpolate_mul_nu1
mul_nu1=interpolate_mul_nu1(base_dir='..')

which_multipoles = ["monopole", "dipole", "quadrupole", "hexadecapole"]


mySignal = Signal(	  BG, 
                      myPert, 
                      mul_nu1, 
                      which_multipoles= which_multipoles, 
                      zin=10, 
                      )

```


## Citation

If using this software, please cite this repository and the paper ''Gravitational Redshift Constraints on the Effective Theory of Interacting Dark Energy'', [arXiv:2311.14425](<https://arxiv.org/abs/2311.14425>). Bibtex:

```
@article{Castello:2023zjr,
    author = "Castello, Sveva and Mancarella, Michele and Grimm, Nastassia and Blanco, Daniel Sobral and Tutusaus, Isaac and Bonvin, Camille",
    title = "{Gravitational Redshift Constraints on the Effective Theory of Interacting Dark Energy}",
    eprint = "2311.14425",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "11",
    year = "2023"
}
```