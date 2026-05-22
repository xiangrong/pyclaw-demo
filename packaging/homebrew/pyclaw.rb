class Pyclaw < Formula
  include Language::Python::Virtualenv

  desc "Python AI Agent - OpenClaw inspired personal assistant with Feishu/Lark support"
  homepage "https://github.com/xiangrong/pyclaw-demo"
  url "https://github.com/xiangrong/pyclaw-demo/releases/download/v0.2.0/pyclaw-0.2.0.tar.gz"
  version "0.2.0"
  sha256 "8b8060a686b060bdccbecd755db96241e81620a8a92d55ff09b49dbc902d710d"

  depends_on "python@3.11"

  def install
    venv = virtualenv_create(libexec, "python3.11")
    venv.pip_install resources
    venv.pip_install buildpath

    # 创建 pyclaw 命令
    (bin/"pyclaw").write_env_script(
      libexec/"bin/pyclaw",
      PYCLAW_HOME: "$HOME/.pyclaw",
    )
  end

  test do
    assert_match "PyClaw", shell_output("#<built-in function bin>/pyclaw --help")
  end
end
