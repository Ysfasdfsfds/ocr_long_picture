#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出架构图为图片文件
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def export_architecture_diagram():
    """导出架构图为PNG图片"""
    
    # 获取当前目录下的HTML文件路径
    html_file = os.path.abspath("architecture_diagram.html")
    
    if not os.path.exists(html_file):
        print("错误：找不到 architecture_diagram.html 文件")
        return
    
    # 配置Chrome选项
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 无头模式
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    
    try:
        # 启动浏览器
        print("正在启动浏览器...")
        driver = webdriver.Chrome(options=chrome_options)
        
        # 打开HTML文件
        file_url = f"file:///{html_file.replace(os.sep, '/')}"
        print(f"正在加载页面: {file_url}")
        driver.get(file_url)
        
        # 等待页面完全加载
        print("等待页面加载完成...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "architecture-stack"))
        )
        
        # 额外等待确保所有样式和动画加载完成
        time.sleep(3)
        
        # 设置窗口大小以适应内容
        driver.set_window_size(1920, driver.execute_script("return document.body.scrollHeight"))
        
        # 截取整个页面
        print("正在截取页面...")
        screenshot_path = "长图OCR系统架构图.png"
        driver.save_screenshot(screenshot_path)
        
        print(f"✅ 架构图已成功导出为: {screenshot_path}")
        print(f"📁 文件位置: {os.path.abspath(screenshot_path)}")
        
        # 获取文件大小
        file_size = os.path.getsize(screenshot_path) / 1024 / 1024  # MB
        print(f"📊 文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        print("\n可能的解决方案:")
        print("1. 确保已安装 Chrome 浏览器")
        print("2. 安装 selenium: pip install selenium")
        print("3. 下载 ChromeDriver 并添加到 PATH")
        print("4. 或者使用手动截图方式")
        
    finally:
        try:
            driver.quit()
        except:
            pass

def manual_export_guide():
    """提供手动导出指南"""
    print("\n📋 手动导出指南:")
    print("1. 在浏览器中打开 http://localhost:8000/architecture_diagram.html")
    print("2. 按 F12 打开开发者工具")
    print("3. 按 Ctrl+Shift+P 打开命令面板")
    print("4. 输入 'screenshot' 选择 'Capture full size screenshot'")
    print("5. 图片将自动下载到默认下载文件夹")

if __name__ == "__main__":
    print("🎨 长图OCR系统架构图导出工具")
    print("=" * 50)
    
    try:
        export_architecture_diagram()
    except ImportError:
        print("❌ 缺少 selenium 库")
        print("请运行: pip install selenium")
        print("然后下载 ChromeDriver: https://chromedriver.chromium.org/")
        manual_export_guide()
    except Exception as e:
        print(f"❌ 自动导出失败: {e}")
        manual_export_guide()