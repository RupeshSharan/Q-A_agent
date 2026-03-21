import urllib.request
import urllib.error

try:
    response = urllib.request.urlopen('http://localhost:8000/api/health')
    print("STATUS:", response.status)
    print("BODY:", response.read().decode())
except urllib.error.HTTPError as e:
    print("HTTP ERROR:", e.code)
    print("BODY:", e.read().decode())
except Exception as e:
    print("EXCEPTION:", str(e))
