#!/usr/bin/env python3
import os
import sys
import subprocess
import webbrowser
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception as e:
    print("tkinter not available: ", e)
    raise


HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8000
    ollama_host: str = "http://127.0.0.1:11434"
    index_dir: str = os.path.join(HERE, "kb")
    input_path: str = os.path.join(HERE, "sample_docs")


class ServerProcess:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None

    def start(self, cfg: Config) -> None:
        if self.proc and self.proc.poll() is None:
            return
        env = os.environ.copy()
        if cfg.ollama_host:
            env["OLLAMA_HOST"] = cfg.ollama_host
        cmd = [sys.executable, "-m", "uvicorn", "api:app", "--host", cfg.host, "--port", str(cfg.port)]
        self.proc = subprocess.Popen(cmd, cwd=HERE, env=env)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ollama KB 启动器")
        self.geometry("640x360")
        self.resizable(False, False)

        self.cfg = Config()
        self.server = ServerProcess()

        pad = 8
        frm = ttk.Frame(self, padding=pad)
        frm.pack(fill=tk.BOTH, expand=True)

        # Fields
        self.var_host = tk.StringVar(value=self.cfg.host)
        self.var_port = tk.StringVar(value=str(self.cfg.port))
        self.var_ollama = tk.StringVar(value=self.cfg.ollama_host)
        self.var_index = tk.StringVar(value=self.cfg.index_dir)
        self.var_input = tk.StringVar(value=self.cfg.input_path)

        def row(parent, r, label, var, browse: Optional[str] = None):
            ttk.Label(parent, text=label, width=18, anchor=tk.W).grid(row=r, column=0, sticky=tk.W, padx=(0,4), pady=4)
            ent = ttk.Entry(parent, textvariable=var)
            ent.grid(row=r, column=1, sticky=tk.EW, padx=(0,4), pady=4)
            if browse:
                def pick():
                    if browse == 'dir':
                        path = filedialog.askdirectory(initialdir=var.get() or HERE)
                    else:
                        path = filedialog.askopenfilename(initialdir=var.get() or HERE)
                    if path:
                        var.set(path)
                ttk.Button(parent, text="选择", command=pick).grid(row=r, column=2, padx=(0,4), pady=4)

        grid = ttk.Frame(frm)
        grid.pack(fill=tk.X)
        grid.columnconfigure(1, weight=1)

        row(grid, 0, "后端 Host", self.var_host)
        row(grid, 1, "后端 Port", self.var_port)
        row(grid, 2, "Ollama 地址", self.var_ollama)
        row(grid, 3, "知识库目录 index_dir", self.var_index, browse='dir')
        row(grid, 4, "本地 sample_docs", self.var_input, browse='dir')

        # Buttons
        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(12,0))

        # Larger, full-width clickable buttons
        style = ttk.Style(self)
        try:
            style.configure('Big.TButton', padding=(16, 12), font=('Helvetica', 13, 'bold'))
        except Exception:
            style.configure('Big.TButton', padding=(16, 12))

        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)

        self.btn_start = ttk.Button(btns, text="启动后端并打开前端", command=self.on_start, style='Big.TButton', width=24)
        self.btn_start.grid(row=0, column=0, sticky='nsew', padx=(0,8))

        self.btn_stop = ttk.Button(btns, text="停止后端", command=self.on_stop, style='Big.TButton', width=24)
        self.btn_stop.grid(row=0, column=1, sticky='nsew', padx=(0,8))

        self.btn_open = ttk.Button(btns, text="仅打开前端", command=self.on_open, style='Big.TButton', width=24)
        self.btn_open.grid(row=0, column=2, sticky='nsew')

        self.status = ttk.Label(frm, text="状态: 未启动")
        self.status.pack(fill=tk.X, pady=(12,0))

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def current_cfg(self) -> Config:
        try:
            port = int(self.var_port.get().strip())
        except Exception:
            port = 8000
        return Config(
            host=self.var_host.get().strip() or "127.0.0.1",
            port=port,
            ollama_host=self.var_ollama.get().strip(),
            index_dir=self.var_index.get().strip(),
            input_path=self.var_input.get().strip(),
        )

    def on_start(self) -> None:
        cfg = self.current_cfg()
        try:
            self.server.start(cfg)
            self.status.config(text="状态: 后端已启动，正在打开浏览器...")
            # open browser with defaults via URL params
            url = f"http://{cfg.host}:{cfg.port}/ui/?ollama_host={quote(cfg.ollama_host)}&index_dir={quote(cfg.index_dir)}&input_path={quote(cfg.input_path)}"
            threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True).start()
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            self.status.config(text="状态: 启动失败")

    def on_stop(self) -> None:
        try:
            self.server.stop()
            self.status.config(text="状态: 已停止")
        except Exception as e:
            messagebox.showwarning("停止异常", str(e))

    def on_open(self) -> None:
        cfg = self.current_cfg()
        url = f"http://{cfg.host}:{cfg.port}/ui/?ollama_host={quote(cfg.ollama_host)}&index_dir={quote(cfg.index_dir)}&input_path={quote(cfg.input_path)}"
        webbrowser.open(url)

    def on_close(self) -> None:
        self.on_stop()
        self.destroy()


def quote(s: str) -> str:
    try:
        from urllib.parse import quote as _q
        return _q(s or "")
    except Exception:
        return s


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


