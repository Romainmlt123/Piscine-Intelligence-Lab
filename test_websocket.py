import asyncio
import aiohttp
import sys

async def test_websocket():
    url = "ws://localhost:8001/ws/audio"
    print(f"Connecting to {url}...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                print("Connected!")
                await ws.send_str("Hello")
                msg = await ws.receive()
                print(f"Received: {msg.data}")
                await ws.close()
                print("Connection closed.")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_websocket())
