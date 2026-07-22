"""调用灵感鸭 API 站点的 gpt-5.5 模型。"""

import os
import sys

import requests


API_URL = "https://www.lingganyaapi.com/v1/chat/completions"
API_KEY = "sk-jWVF88xbin6KHB4hpyVrO7SWCZq8CxQ6VewPY6rzst4HeJXY"


def main() -> int:
    api_key = API_KEY

    user_message = " ".join(sys.argv[1:]).strip()
    if not user_message:
        user_message = input("请输入要发送给 gpt-5.5 的问题：").strip()
    if not user_message:
        print("错误：问题不能为空。")
        return 1

    payload = {
        "model": "gpt-5.6-luna",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业、准确的中文助手。",
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=(10, 180),
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
    except requests.HTTPError:
        print(f"请求失败，HTTP 状态码：{response.status_code}")
        print(response.text)
        return 1
    except requests.RequestException as exc:
        print(f"网络请求失败：{exc}")
        return 1
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"响应格式解析失败：{exc}")
        print(response.text)
        return 1

    print("\ngpt-5.5 回复：")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
