# -*- coding: utf-8 -*-
"""Novel Constellation 一键启动器：托管后端、静态服务器和浏览器。"""

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
PYTHON_EXE = sys.executable
BACKEND_URL = "http://127.0.0.1:5001/api/health"
FRONTEND_URL = "http://127.0.0.1:8080/"


def url_ready(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_service(name, url, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if url_ready(url):
            print(f"  {name}已就绪")
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{name}进程提前退出（代码 {process.returncode}）")
        time.sleep(0.25)
    raise RuntimeError(f"等待{name}超时，请查看上方日志")


def start_process(command, log_name):
    """启动与启动器窗口解耦的后台服务，避免窗口关闭后页面断开。"""
    flags = 0
    if os.name == "nt":
        flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    log_path = os.path.join(BASE_DIR, log_name)
    log_file = open(log_path, "ab", buffering=0)
    try:
        return subprocess.Popen(
            command,
            cwd=BASE_DIR,
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    finally:
        log_file.close()


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def validate_files():
    required = [
        os.path.join(BACKEND_DIR, "app.py"),
        os.path.join(BACKEND_DIR, "requirements.txt"),
        os.path.join(BASE_DIR, "index.html"),
        os.path.join(BASE_DIR, "assets", "app.js"),
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise RuntimeError("项目文件不完整：\n  " + "\n  ".join(missing))
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.isfile(env_path):
        raise RuntimeError("未找到 .env，请复制 .env.example 并填写 LLM_API_KEY")

    api_key = ""
    with open(env_path, "r", encoding="utf-8-sig") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "LLM_API_KEY":
                api_key = value.strip().strip('"').strip("'")
                break

    placeholder_values = {"你的_API_KEY", "your_api_key", "your-api-key", "changeme"}
    if not api_key or api_key.lower() in {value.lower() for value in placeholder_values}:
        raise RuntimeError("请先打开 .env，填写有效的 LLM_API_KEY")


def main():
    backend_process = None
    frontend_process = None

    print("=" * 56)
    print("  Novel Constellation · 小说人物关系 3D 星云")
    print("=" * 56)

    try:
        validate_files()

        if url_ready(BACKEND_URL):
            print("[1/2] 后端已在运行，直接复用")
        else:
            if port_in_use(5001):
                raise RuntimeError("端口 5001 已被其他程序占用")
            print("[1/2] 正在启动后端 http://127.0.0.1:5001 ...")
            backend_process = start_process(
                [PYTHON_EXE, "-u", os.path.join(BACKEND_DIR, "app.py")],
                "backend-service.log",
            )
            wait_for_service("后端", BACKEND_URL, backend_process, 20)

        if url_ready(FRONTEND_URL):
            print("[2/2] 前端已在运行，直接复用")
        else:
            if port_in_use(8080):
                raise RuntimeError("端口 8080 已被其他程序占用")
            print("[2/2] 正在启动前端 http://127.0.0.1:8080 ...")
            frontend_process = start_process(
                [PYTHON_EXE, "-u", "-m", "http.server", "8080", "--bind", "127.0.0.1"],
                "frontend-service.log",
            )
            wait_for_service("前端", FRONTEND_URL, frontend_process, 10)

        print("=" * 56)
        print("  启动成功：" + FRONTEND_URL)
        print("  前后端已在后台运行，现在可以关闭此窗口。")
        print("=" * 56)

        if os.getenv("MING3D_NO_BROWSER") != "1":
            webbrowser.open(FRONTEND_URL)
        time.sleep(1)
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\n启动已取消。")
        stop_process(frontend_process)
        stop_process(backend_process)
        return 1
    except Exception as exc:
        print(f"\n[启动失败] {exc}")
        # 只有启动失败时才清理本次新建的进程，避免留下半启动状态。
        stop_process(frontend_process)
        stop_process(backend_process)
        if os.getenv("MING3D_TEST_EXIT") != "1":
            input("按回车键退出...")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
