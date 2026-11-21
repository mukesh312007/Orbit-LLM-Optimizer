# Orbit benchmark methodology

Benchmark configurations run sequentially so local model RAM is not shared by
simultaneous inference servers. Warm-up generations are recorded separately
from measured trials. Spider text-to-SQL quality uses exact SQL comparison and
normalized SQL exact comparison. When the gold SQL contains `ORDER BY`,
the result row sequence must match in order.

Measured performance includes end-to-end latency, time to first token, output
tokens per second, and peak process RAM. CPU utilisation can produce an energy
proxy, but it is not a hardware energy measurement.
