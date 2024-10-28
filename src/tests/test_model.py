import os

from click.testing import CliRunner

from ftw_cli.cli import model_fit, model_test, data_download

CKPT_FILE = "logs/FTW-CI/lightning_logs/version_0/checkpoints/last.ckpt"
CONFIG_FILE = "src/tests/data-files/min_config.yaml"

def test_model_fit():
    runner = CliRunner()

    # Check help
    result = runner.invoke(model_fit, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: fit [OPTIONS] [CLI_ARGS]..." in result.output

    # Download required data for the fit command
    runner.invoke(data_download, ["--countries=Kenya,Rwanda"])
    assert os.path.exists("data/ftw/kenya")
    assert os.path.exists("data/ftw/rwanda")
    assert os.path.exists(CONFIG_FILE)
    
    # Run minimal fit
    result = runner.invoke(model_fit, ["-c", CONFIG_FILE])
    assert result.exit_code == 0, result.output
    assert "Train countries: ['kenya', 'rwanda']" in result.output
    assert "Epoch 0: 100%|" in result.output
    assert "`Trainer.fit` stopped: `max_epochs=1` reached." in result.output
    assert os.path.exists(CKPT_FILE)

def test_model_test():
    runner = CliRunner()

    # Check help
    result = runner.invoke(model_test, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: test [OPTIONS] [CLI_ARGS]..." in result.output

    # Actually run the test
    result = runner.invoke(model_test, [
        "--gpu", "0",
        "--model", CKPT_FILE,
        "--countries", "Kenya", # should be "kenya", but let's test case insensitivity
        "--out", "results.csv"
    ])
    assert result.exit_code == 0, result.output
    assert "Running test command" in result.output
    assert "Created dataloader" in result.output
    assert "100%|" in result.output
    assert "Object level recall: 0.0000" in result.output
    assert os.path.exists("results.csv")

    # TODO: Add more tests
