# Optimization algorithm

Candidates must have repeated benchmark evidence. Hard constraints are applied first: conservative quality lower-confidence bound, maximum latency, RAM, and actual physical energy. Missing requested values fail a constraint rather than being guessed.

After the quality gate, the documented objective is deterministic rather than a hidden weighted score: latency/balanced minimizes measured latency, then RAM, then maximizes throughput; memory minimizes RAM first; quality maximizes quality first; energy-first requires comparable physical joule readings. CPU-utilisation energy proxies may be displayed for exploratory analysis but never satisfy an energy constraint or rank candidates automatically. A non-dominated quality/latency/RAM Pareto frontier and explicit rejected alternatives are returned.
