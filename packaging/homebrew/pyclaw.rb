class Pyclaw < Formula
  include Language::Python::Virtualenv

  desc "Python AI Agent - OpenClaw inspired personal assistant with Feishu/Lark support"
  homepage "https://github.com/xiangrong/pyclaw-demo"
  url "file:///Users/mac/.hermes/hermes-agent/pyclaw-demo/dist/pyclaw-0.1.0.tar.gz"
  version "0.1.0"
  sha256 "85e8427ab9c2b6dca7bf6a4b6754b092755298ac813d7293f94c174551406b1d"

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
