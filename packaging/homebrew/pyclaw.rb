class Pyclaw < Formula
  include Language::Python::Virtualenv

  desc "Python AI Agent - OpenClaw inspired personal assistant with Feishu/Lark support"
  homepage "https://github.com/xiangrong/pyclaw-demo"
  url "https://github.com/xiangrong/pyclaw-demo/releases/download/v0.1.1/pyclaw-0.1.1.tar.gz"
  version "0.1.1"
  sha256 "60f75f013d7c43eca12ac0883fc68dfa67b938f7c8f2d9ea620ba78d077ab7ab"

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
