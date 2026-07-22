import http.client
import json

#    "model": "gemini-3.1-flash-lite",
#    "model": "gpt-5.4",

conn = http.client.HTTPSConnection("https://www.lingganyaapi.com/pricing")
payload = json.dumps({
   "model": "gpt-5.5",
   "stream": False,
   "messages": [
      {
         "role": "user",
         "content": "你好"
      }
   ]
})
headers = {
   'Authorization': 'sk-jWVF88xbin6KHB4hpyVrO7SWCZq8CxQ6VewPY6rzst4HeJXY',
   'Content-Type': 'application/json'
}
conn.request("POST", "/v1/chat/completions", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))