# Orbit local telemetry

JSONL experiment files are the durable local benchmark evidence and remain the
fallback if MySQL is unavailable. When configured, local MySQL mirrors live
selection and full-matrix benchmark telemetry. The MySQL service and database
are local; database failures must not prevent a local answer or JSONL logging.
