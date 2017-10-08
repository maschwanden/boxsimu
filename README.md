# boxsimu

boxsimu is a simple simulation software that allows the 
user to model/simulate boxmodel-systems of intermediate complexity.
It offers a user-friendly interface to define a system by instantiating 
different classes like ```Fluid```, ```Variable```, ```Flow```, ```Flux```, 
```Process```, ```Reaction```, ```Box```, ```BoxModelSystem```. These instances 
can then be connected to each other. Once a system is
defined with boxsimu, its temporal evolution can easily be simulated 
using the ```solve()``` function.

## Getting Started

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

boxsimu can be installed using pip. On your console type in

```pip install boxsimu```

this should automatically compile all Cython files and copy all source 
files into the right directory. If the installation fails check if all 
of the above mentioned dependencies are met.

### Code Example
The system we want to model consists of a freshwater lake that only has 
one inflow and one outflow. We want to simulate how the concentration 
of phosphate in this lake evolves over time. The system is defined 
in boxsimu with the following code:

```python
import boxsimu 
from boxsimu import ur

freshwater = boxsimu.Fluid('freshwater', rho=1000*ur.kg/ur.meter**3)
po4 = boxsimu.Variable('po4')

lake = boxsimu.Box(
    name='lake',
    description='Little Lake',
    fluid=freshwater.q(m_water),
    variables=[po4.q(m_0)],
)

inflow = boxsimu.Flow(
    name='Inflow', 
    source_box=None,
    target_box=lake,
    rate=flow_rate,
    tracer_transport=True,
    concentrations={po4: 3e-1 * ur.gram / ur.kg}, 
)
outflow = boxsimu.Flow(
    name='Outflow',
    source_box=lake,
    target_box=None,
    rate=flow_rate,
    tracer_transport=True,
)

system = boxsimu.BoxModelSystem(
    name='lake_system', 
    description='Simple Lake Box Model',
    boxes=[lake,], 
    flows=[inflow, outflow,],
)
```
For an explanation of the code see *Tutorial Part 1*.
The system's temporal evolution can then be simulated and visualized:

```python
solution = system.solve(total_integration_time=800*ur.day, dt=1*ur.day)
solution.plot_variable_concentration(variable=po4, figsize=(6,4), units=ur.kg/ur.meter**3)
```
This results in the following plot:
![PO4 concentration as a function of time](https://github.com/maschwanden/boxsimu/raw/master/img/tutorial1_simulation_po4_plot.png)

## Authors

* **Mathias Aschwanden** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details






