# GPT-5.5 Python 调用脚本使用说明

桌面脚本 `call_gpt55.py` 用于调用灵感鸭 API 站点的 `gpt-5.5` 模型。

## 一、准备环境

电脑需要安装 Python 3。打开 PowerShell，检查：

```powershell
python --version
```

安装脚本依赖：

```powershell
python -m pip install requests
```

## 二、设置 API 密钥

使用用户自己在站内创建的 API 密钥。只对当前 PowerShell 窗口临时设置：

```powershell
$env:LINGGANYA_API_KEY="你的站内API密钥"
```

关闭当前 PowerShell 窗口后，该临时环境变量会自动失效。

不要把真实 API 密钥直接写入 Python 脚本，也不要把密钥发送给其他人或提交到 Git 仓库。

## 三、运行脚本

进入桌面：

```powershell
cd "$HOME\Desktop"
```

方式一：运行后再输入问题：

```powershell
python .\call_gpt55.py
```

看到下面的提示后输入问题并按回车：

```text
请输入要发送给 gpt-5.5 的问题：
```

方式二：直接在命令中携带问题：

```powershell
python .\call_gpt55.py "请用简单的语言介绍人工智能"
```

## 四、完整操作示例

```powershell
cd "$HOME\Desktop"
python -m pip install requests
$env:LINGGANYA_API_KEY="你的站内API密钥"
python .\call_gpt55.py "你好，请介绍一下你自己"
```

## 五、常见错误

### 提示没有设置 LINGGANYA_API_KEY

先执行：

```powershell
$env:LINGGANYA_API_KEY="你的站内API密钥"
```

必须在同一个 PowerShell 窗口中继续运行 Python 脚本。

### 提示 No module named requests

执行：

```powershell
python -m pip install requests
```

### 返回 401

通常表示 API 密钥无效、填写错误或已经失效，请重新复制用户在站内创建的密钥。

### 返回 403

通常表示当前密钥或用户分组没有 `gpt-5.5` 模型权限。

### 返回 503 或没有可用渠道

表示当前模型暂时没有可用渠道，可以稍后重试或联系管理员检查渠道状态。

### 请求超时

检查本机网络后重新运行。如果模型生成内容较长，响应时间可能相应增加。
