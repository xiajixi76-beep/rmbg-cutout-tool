@echo off
chcp 65001 >nul
cd /d E:\cutout_tool
set PORT=7861
echo 正在启动本地抠图工具（RMBG-2.0）...
echo 浏览器会自动打开 http://127.0.0.1:7861
echo 关闭此窗口即停止服务。
venv\Scripts\python.exe app.py
pause
