from knowledgelens import cli


class _FakeDatabase:
    enabled = True

    def initialize(self):
        pass


def test_bootstrap_admin_cli_prompts_twice_and_never_accepts_password_argument(monkeypatch, capsys):
    parser = cli.build_parser()
    try:
        parser.parse_args(["bootstrap-admin", "--username", "admin", "--password", "secret"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("password must never be accepted on the command line")

    monkeypatch.setattr(cli, "Database", _FakeDatabase)
    answers = iter(["one password", "another password"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: next(answers))
    assert cli.main(["bootstrap-admin", "--username", "admin"]) == 2
    assert "Passwords do not match" in capsys.readouterr().err


def test_bootstrap_admin_cli_creates_first_admin(monkeypatch, capsys):
    monkeypatch.setattr(cli, "Database", _FakeDatabase)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: "correct horse battery staple")
    captured = {}

    def fake_bootstrap(_database, username, password):
        captured["username"] = username
        captured["password"] = password
        return True

    monkeypatch.setattr(cli, "bootstrap_admin", fake_bootstrap)
    assert cli.main(["bootstrap-admin", "--username", "admin"]) == 0
    assert captured == {"username": "admin", "password": "correct horse battery staple"}
    assert "Created bootstrap administrator" in capsys.readouterr().out
