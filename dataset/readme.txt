included files:
- ts_mon.pkl
- ts_unmon.pkl

ts_mon.pkl contains a dictionary that contains split samples for each of the 95 websites in the monitored set. The number of samples per class were clipped to 200.
ts_unmon.pkl contains all the collected unmonitored samples in one large list. You will probably want to clip with count down to approximately the same size as your monitored set.

All traffic samples are represented in the timestamp*direction format. If you need to recover only the cell timestamp, just take the absolute value of the value. If you need to recover only the direction, just take signage of value (e.g. +1 outgoing, -1 incoming). Treat timestamps of 0 (no signage value) as +1 as this should only appear as the first packet in the trace.

Email me (njm3308@rit.edu) if you have questions, encounter issues, or otherwise need help.