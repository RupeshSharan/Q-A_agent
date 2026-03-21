import urllib.request
import urllib.error

endpoints = ['/api/auth/me', '/api/health']

for endpoint in endpoints:
    try:
        req = urllib.request.Request('http://localhost:8000' + endpoint)
        req.add_header('Authorization', 'Bearer invalidtoken')
        response = urllib.request.urlopen(req)
        print(f"{endpoint} STATUS: {response.status}")
    except urllib.error.HTTPError as e:
        print(f"{endpoint} HTTP ERROR: {e.code}")
        print(f"{endpoint} BODY: {e.read().decode()}")
    except Exception as e:
        print(f"{endpoint} EXCEPTION: {str(e)}")
