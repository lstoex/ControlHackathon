Repository for the Hackathon at the `Future PhD in Control' in Hannover, April 2025.

---
In groups from 2-4. the students work on a self-chosen optimal control problem.

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
	
For every of the models and the methods, we prepared examples, that you can use as a starting point.


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