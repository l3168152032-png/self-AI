import os

# 可选：通过环境变量注入 DLL 搜索路径（避免写死到你本机的盘符/目录）
_dll_dir = os.environ.get("NEURO_DLL_DIR")
if _dll_dir and os.path.exists(_dll_dir):
    os.add_dll_directory(_dll_dir)
import subprocess
import time
import sys

def start_neuro():
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    body_script = os.path.join(REPO_ROOT, "src", "body", "neuro_body.py")
    brain_script = os.path.join(REPO_ROOT, "src", "core", "neuro_brain.py")
    print("🌟 Neuro 全系统启动序列开始...")

    # 1. 启动身体控制中心 (neuro_body.py)
    # 使用 subprocess.Popen 异步启动，不阻塞主进程
    print("📡 正在唤醒身体控制中心...")
    body_process = subprocess.Popen([sys.executable, body_script], cwd=REPO_ROOT)
    
    # 稍微等 2 秒，确保 VTS 连接成功后再启动大脑
    time.sleep(2)

    # 2. 启动大脑推理中心 (neuro_brain.py)
    print("🧠 正在加载 Neuro 的深度学习大脑...")
    try:
        # 使用 run 启动大脑，因为它需要我们在终端输入【你】: xxx
        # 这会占用当前的终端窗口
        subprocess.run([sys.executable, brain_script], cwd=REPO_ROOT)
    except KeyboardInterrupt:
        print("\n\n👋 正在关闭 Neuro 系统...")
    finally:
        # 3. 当大脑关闭时，自动把身体进程也杀掉，防止后台残留
        body_process.terminate()
        print("✅ 系统已安全退出。")

if __name__ == "__main__":
    start_neuro()