# FCP_summative

Link to github repository:
https://github.com/3llag0dek/fcp_summative.git

Following imports are needed: (for tasks 1, 3 and 4)
- numpy as np
- matplotlib.pyplot as plt
- matplotlib.cm as cm
- argparse
- random

Following imports are needed: (for tasks 2 and 5)
- argparse
- random
- numpy as np
- matplotlib.pyplot as plt
- unittest

The location of files Task_1_3_4.py, Task_2.py and Task_5.py must be known, they should all be in the same zipped folder. 
Using the terminal access the file Task_1_3_4.py for tasks 1, 3 and 4.
Using the terminal access the file Task_2.py for task 2.
Using the terminal access the file Task_5.py for task 5.

Running task 1:
python task_1_3_4.py -ising_model -external<the strength of external opinion>-alpha<probability of fliping opinion>
(this should plot an animation for changing population opinion)
python task_1_3_4.py -test_ising
(this should test the code and return :
"tests passed" if there is no mistake in calculating the agreement
"test7-10" if there is an error with the external variable in calculating the agreement )
else: a mistake in the equation)

Running task 3:
python task_1_3_4.py -network <N> 
  (this should create and plot a random network of size N, and return:
  Mean degree: <number>
  Average path length: <number>
  Clustering co-efficient: <number>)
python task_1_3_4.py -test_network
  (this should run the provided test functions and return:
  'All Tests Passed' if it is successful)

Running task 4:
python task_1_3_4.py -ring_network <number of nodes>
  (this should output a ring network with the amount of nodes entered by the user)
python task_1_3_4.py -small_world <number of nodes>
  (this should output a small world network with the amount of nodes entered by the user)
python task_1_3_4.py -small_world <number of nodes> -re_wire <re-wire probability>
  (this should output a small world network with the amount of nodes and rewire probability entered by the user)
