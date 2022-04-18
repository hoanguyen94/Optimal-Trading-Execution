Process: 
1. Run NASDAQParser.py to parse raw data
2. Run MDBookHandler.py to build LOB and extract files
3. Run DataAnalysis.py to proprocess data and save optimal policies, save it as csv file
4. Run Training.py to train the model with Q-learning (with or without deterministic experience replay)
5. For stock CATY, run Training - Replay.py to train the model with Q-learning with random experience replay
6. Run Benchmark.py to estimate performance of benchmark strategies
7. Run Performance.py to extract the results of Q-learning policies with learning rate 0.1

NOTE: 
- For stock CATY, if you already run Q-learning with learning rate 0.1, 0.3, and with deterministic experience
replay and learning rate 0.1-0.1, 0.1-0.3, run ConvergencePlot.ipynb to collect performance results and convergence plots
for standard Q-learning and Q-learning with deterministic replay.
(Be aware of the file names and change them to the ones you saved)

- If you already run Q-learning with random experience replay (both uniform and prioritized), run ConvergencePlot-Replay.ipynb
to collect performance results and convergence plots for Q-learning with random experience replay. 
(Be aware of the file names and change them to the ones you saved)

6. Run SimulationFinal.py to simulate data
7. Run Training - Sim.py to train the model with simulated data
6. Run Benchmark.py to estimate performance of benchmark strategies
7. Run Performance.py to extract the results of Q-learning policies with learning rate 0.1