# boxsimu

boxsimu is a simple simulation software that allows the 
user to model/simulate simple to intermediate complex boxmodel-systems.
It offers a user friendly interface to generate boxes, processes and 
reactions that take place inside these boxes, and flows/fluxes that 
describe how boxes exchange fluid and variable mass. Once a system is
defined with boxsim, its temporal resolution can easily be solved using 
the ```solve()``` function.

## Getting Started

boxsimu is available in the official python repository and thus can be
installed using "pip install boxsimu".


### Prerequisites
**Scientifc packages:**<br>
- **numpy** (1.10 or newer)
- **matplotlib** (1.5 or newer)
- **jupyter** (4.0 or newer)
- **pandas** (0.15 or newer)
- **pint** (0.8.1 or newer)

**Other packages:**<br>
- **attrdict** (1.2 or newer)
- **svgwrite** (1.1 or newer)
- **dill** (2.7 or newer)

### Installing

boxsimu can easily be installed using pip. On your console type in 
> $ pip install boxsimu

this should automatically compile all Cython files and copy all source 
files into the right directory. If the installation fails check if all 
of the above mentioned dependencies are met.

## Authors

* **Mathias Aschwanden** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details






