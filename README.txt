notes on chemical process
=========================

- there are 7 components
  - f1, f2, purge, product, tank, composition, plc
  - the plc reads sensor and actuator values from the other components and
    calculates an updated state
  - all sensor and actuators are limited to a range of [0-65535]
  - the dataset is limited variables collected from the plc

- f1
  - f1 is the main process input
  - f1 is a valve containing one sensor and one actuator
    - sensor  : the flow rate of f1
    - actuator: determines position of valve (i.e., how far to open it) where
                0 is all the way closed and 65535 is all the way open

- f2
  - f2 is the secondary process input
  - f2 is a valve containing one sensor and one actuator
    - sensor  : the flow rate of f2
    - actuator: determines position of valve (i.e., how far to open it) where
                0 is all the way closed and 65535 is all the way open
- purge
  - purge acts as a pressure override when the pressure in the tank grows too
    high
  - purge is a valve containing one sensor and one actuator
    - sensor  : the output flow rate of purge
    - actuator: determines position of valve (i.e., how far to open it) where
                0 is all the way closed and 65535 is all the way open
- product
  - product is the main output for the process
  - product is a valve containing one sensor and one actuator
    - sensor  : the output flow rate of product
    - actuator: determines position of valve (i.e., how far to open it) where
                0 is all the way closed and 65535 is all the way open

- tank
  - the tank is where the chemical reaction takes place. it receives input
    from f1 and f2 and outputs product to product and waste to purge.
  - the tank has two sensors
    - pressure
    - liquid level

- composition
  - the composition determines the percentage of each chemical which has been
    purged.
  - composition has 3 sensors
    - percentage of A in purge
    - percentage of B in purge
    - percentage of C in purge


dataset field definitions
=========================

- time     : time elapsed since beginning of dataset

- f1_c     : control calculated by plc to write to f1 valve
- f1_a     : actuator value (valve position) of f1 received by the plc
- f1_s     : sensor value of f1 (flow rate) received by the plc
- f1_d     : difference in f1_s from previous reading

- f2_c     : control calculated by plc to write to f2 valve
- f2_a     : actuator value of f1 (valve position) received by the plc
- f2_s     : sensor value of f1 (flow rate) received by the plc
- f2_d     : difference in f2_s from previous reading

- prg_c    : control calculated by plc to write to purge valve
- prg_a    : actuator value (valve position) of purge received by the plc
- prg_s    : sensor value (flow rate) of purge received by plc
- prg_d    : difference in prg_s from previous reading

- prd_c    : control calculated by plc to write to product value
- prd_a    : actuator value (valve position) of product received by the plc
- prd_s    : sensor value (flow rate) of product received by the plc
- prd_d    : difference in prd_s from previous reading

- pr_s     : sensor value (pressure) received by plc from the tank
- pr_d     : difference in pr_s from previous reading

- lq_s     : sensor value (liquid level) received by plc from the tank
- lq_d     : difference in lq_s from previous reading

- cmp_a_s  : sensor value (% of chemical A that's been purged) received by plc
- cmp_a_d  : difference in cmp_a_s from previous reading

- cmp_b_s  : sensor value (% of chemical B that's been purged) received by plc
- cmp_b_d  : difference in cmp_b_s from previous reading

- cmp_c_s  : sensor value (% of chemical C that's been purged) received by plc
- cmp_c_d  : difference in cmp_c_s from previous reading
