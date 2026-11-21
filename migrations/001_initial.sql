-- Apply with a MySQL account that can create tables after creating database llm_optimizer.
-- The authoritative schema is database/models.py; this migration establishes version tracking.
CREATE TABLE IF NOT EXISTS schema_migrations (version VARCHAR(32) PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
INSERT IGNORE INTO schema_migrations(version) VALUES ('001_initial');
