import app


class _FakeDemo:
    def __init__(self):
        self.launch_kwargs = None

    def queue(self, **_kwargs):
        return self

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs


def test_main_defaults_to_all_interfaces(monkeypatch):
    fake_demo = _FakeDemo()

    monkeypatch.setattr(app, "initialize", lambda root: ((object(), [])))
    monkeypatch.setattr(app, "build_ui", lambda state, messages: fake_demo)

    assert app.main([]) == 0
    assert fake_demo.launch_kwargs["server_name"] == "0.0.0.0"
