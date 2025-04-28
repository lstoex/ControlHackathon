TODO: Intro Text. Welcome to the hackathon ....
What is the goal of this hackathon? -> Students get a feeling for what kind of tasks/problems they might encounter, see the 'power of control' on simple examples.

---
You can do whatever you like or find interesting, feel free to modify any code you like.

----


### Organisation
Timetable
- Day 1, **17:00 - 18:30**, Introduction to Hackathon, find groups
- Day 2, **8:30**, Project Committments (Group Members and Project Working Title)
- Day 2, **8:30 - 12:30**, Hackathon
- Day 2, **12:30 – 14:15**, Presentation of hackathon results by participants

Groups of 2-4 students, supervised by PhD students and Professors.

### Ideas for Models & Methods

For inspiration, we prepared a few models and methods that you can use as a starting point. 

Ideas for Models:
- [2D-XY Bicycle](documentation/Scenario%20-%202D-XY%20Bicycle%20Control.md)
- [2D-XZ Drone](documentation/Scenario%20-%202D-XZ%20Drone%20Control.md)
- [2D-XZ Rocket](documentation/Scenario%20-%202D-XZ%20Rocket%20Control.md)

Ideas for Methods:
- Open Loop
	- [Open Loop Planning](documentation/Method%20-%20Open%20Loop%20Planning.md)
- Closed Loop
	- [Linear Quadratic Regulator](documentation/Method%20-%20LQR%20Controller.md)
	- [Model Predictive Control](documentation/Method%20-%20Model%20Predictive%20Control.md)
	
For every of the models, we prepared python code that implements:
- the right-hand side of the dynamics, i.e., the function $f(x,u)$
  ```python
  model.dynamics(x, u)
  ```
- a function to compute the linearizations $A = \frac{\partial f}{\partial x} (\bar{x},\bar{u})$ and $B = \frac{\partial f}{\partial u} (\bar{x},\bar{u})$ around a point $\bar{x}, \bar{u}$
  ```python
  A,B = model.linearizedDynamics(x_ss, u_ss)
  ```

- an open-loop simulation (TODO: Picture of open loop system) with a nice visualization. The simulation requires a starting state and an open-loop control-strategy $u(t)$ as inputs.
  ```python
	def u_t(t):
		if t < 3:
			return 0
		else:
			return 1
			
	scenario.openLoopSimulation(x_0, u_t, t_f=10)
	```
- a close-loop simulation:
  ```python
	model.closedLoopSimulation(x_0, u_t, t_f=10)`
	```


### Tools
- Python with standard libraries
	- Numpy
	- Matplotlib
- We use the python framework [`CasAdi`](https://web.casadi.org/) to formulate OCPs
### Installation

1. Install Python and required packages. `numpy, matplotlib, casadi`. If you want you can create a virtual environment first.
2. Clone this repository using `git clone` into a folder of choice
3. Run the example file
	```bash
	python examples/TODO.py
	```
	to see that everything works as intended.