TODO: Intro Text. Welcome to the hackathon ....
What is the goal of this hackathon? -> Students get a feeling for what kind of tasks/problems they might encounter, see the 'power of control' on simple examples.

---
You can do whatever you like or find interesting!

----


### Organisation
Timetable
- Day 1, **17:45-18:15**, Introduction to Hackathon, find groups
- Until Day 2, **8:30**, decide what project you want to work on
- Day 2, **8:30-12:30**, Hackathon
- Day 2, **13:30 – 14:15**, Presentation of hackathon results by participants
Groups of 2-4 students, supervised by PhD students and Professors.

### Tools
- Python with standard libraries
	- Numpy
	- Matplotlib
- We use the python framework [`CasAdi`](https://web.casadi.org/) to formulate OCPs
### Installation

1. Install Python and required packages. If you want you can create a virtual environment first.
2. Clone this repository using `git clone` into a folder of choice
3. 
### Scenarios & Methods

List of Scenarios:
- [Scenario - 2D-XY Bicycle Control](documentation/Scenario%20-%202D-XY%20Bicycle%20Control.md)
- [Scenario - 2D-XZ Drone Control](documentation/Scenario%20-%202D-XZ%20Drone%20Control.md)

List of Methods:
- Open Loop
	- [Method - Open Loop Planning](documentation/Method%20-%20Open%20Loop%20Planning.md)
- Closed Loop
	- [LQR](Method%20-%20LQR%20Controller.md)
	- [Method - Model Predictive Control](documentation/Method%20-%20Model%20Predictive%20Control.md)
	
For every scenario, we prepared python code that implements:
- the model equations
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
	scenario.closedLoopSimulation(x_0, u_t, t_f=10)`
	```
