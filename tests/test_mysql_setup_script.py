from pathlib import Path


def test_mysql_setup_script_prompts_for_root_password_and_preserves_jsonl_logging():
    script = (Path(__file__).parents[1] / "scripts" / "setup-orbit-mysql.ps1").read_text(encoding="utf-8")
    assert "Read-Host \"Enter the local MySQL root password\" -AsSecureString" in script
    assert "database.initialize" in script
    assert "JSONL logging remains enabled" in script
    assert "RandomNumberGenerator]::Create()" in script
