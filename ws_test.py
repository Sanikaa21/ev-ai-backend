import websocket

try:
    ws = websocket.create_connection(
        "ws://127.0.0.1:8000/live",
        timeout=10
    )

    print("CONNECTED TO EVOXIS BACKEND!")

    for i in range(5):
        data = ws.recv()
        print("Received:", data)

    ws.close()

except Exception as e:
    print("WEBSOCKET ERROR:")
    print(e)